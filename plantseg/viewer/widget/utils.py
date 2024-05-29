import timeit
from concurrent.futures import Future
from functools import partial
from typing import Callable, Optional, Tuple

from magicgui.widgets import Widget
from napari import Viewer
from napari.qt.threading import thread_worker

from plantseg.viewer.dag_handler import dag_manager
from plantseg.viewer.logging import napari_formatted_logging


def identity(*args, **kwargs):
    """
    Pass through any positional arguments and ignores any keywords arguments
    """
    if len(args) == 1:
        return args[0]

    elif len(args) > 1:
        return args

    raise ValueError("identity should have at least one positional argument")


def setup_layers_suggestions(viewer: Viewer, out_name: str, widgets: list):
    if out_name not in viewer.layers:
        return None

    out_layer = viewer.layers[out_name]
    for widget in widgets:
        widget.value = out_layer


def start_threading_process(
    func: Callable,
    runtime_kwargs: dict,
    statics_kwargs: dict,
    out_name: str,
    input_keys: Tuple[str, ...],
    layer_kwarg: dict,
    layer_type: str = "image",
    step_name: str = "",
    skip_dag: bool = False,
    viewer: Viewer = None,
    widgets_to_update: list = None,
) -> Future:
    runtime_kwargs.update(statics_kwargs)
    thread_func = thread_worker(partial(func, **runtime_kwargs))
    future = Future()
    timer_start = timeit.default_timer()

    def on_done(result):
        timer = timeit.default_timer() - timer_start
        napari_formatted_logging(f"Widget {step_name} computation complete in {timer:.2f}s", thread=step_name)
        _func = func if not skip_dag else identity
        dag_manager.add_step(
            _func, input_keys=input_keys, output_key=out_name, static_params=statics_kwargs, step_name=step_name
        )
        result = result, layer_kwarg, layer_type
        future.set_result(result)

        if viewer is not None and widgets_to_update is not None:
            setup_layers_suggestions(viewer, out_name, widgets_to_update)

    worker = thread_func()
    worker.returned.connect(on_done)
    worker.start()
    napari_formatted_logging(f"Widget {step_name} computation started", thread=step_name)
    return future


def start_prediction_process(
    func: Callable,
    runtime_kwargs: dict,
    statics_kwargs: dict,
    out_name: str,
    input_keys: Tuple[str, ...],
    layer_kwarg: dict,
    layer_type: str,
    step_name: str,
    skip_dag: bool,
    viewer: Viewer,
    widgets_to_update: Optional[list] = None,
) -> Future:
    assert out_name == layer_kwarg["name"], "out_name and layer_kwarg name should be the same"

    runtime_kwargs.update(statics_kwargs)
    thread_func = thread_worker(partial(func, **runtime_kwargs))
    future = Future()
    timer_start = timeit.default_timer()

    def on_done(result):
        timer = timeit.default_timer() - timer_start
        napari_formatted_logging(f"Widget {step_name} computation complete in {timer:.2f}s", thread=step_name)
        _func = func if not skip_dag else identity

        if result.ndim == 4:  # then we have a 2-channel output, output is always CZYX or ZYX
            pmap_layers = []
            for i, pmap in enumerate(result):
                temp_layer_kwarg = layer_kwarg.copy()
                temp_layer_kwarg["name"] = layer_kwarg["name"] + f"_{i}"
                pmap_layers.append((pmap, temp_layer_kwarg, layer_type))
                dag_manager.add_step(
                    _func,
                    input_keys=input_keys,
                    output_key=temp_layer_kwarg["name"],
                    static_params=statics_kwargs,
                    step_name=step_name,
                )
            result = pmap_layers

            # Only widget_unet_predictions() invokes and handles 4D UNet output for now, but headless mode can also invoke this part, thus warn:
            napari_formatted_logging(
                f"Widget {step_name}: Headless mode is partially supported for 2-channel output predictions.\n"
                "Supported headless workflow: open file -> 2-channel prediction -> save file.\n"
                "More steps following 2-channel prediction are not supported in headless mode.",
                thread=step_name,
                level="warning",
            )
        else:  # then we have a 1-channel output
            result = result, layer_kwarg, layer_type
            dag_manager.add_step(
                _func,
                input_keys=input_keys,
                output_key=layer_kwarg["name"],
                static_params=statics_kwargs,
                step_name=step_name,
            )

        future.set_result(result)

        if viewer is not None and widgets_to_update is not None:
            setup_layers_suggestions(viewer, out_name, widgets_to_update)

    worker = thread_func()
    worker.returned.connect(on_done)
    worker.start()
    napari_formatted_logging(f"Widget {step_name} computation started", thread=step_name)
    return future


def layer_properties(name, scale, metadata: dict = None):
    keys_to_save = {"original_voxel_size", "voxel_size_unit", "root_name"}
    if metadata is not None:
        _new_metadata = {key: metadata[key] for key in keys_to_save if key in metadata}
    else:
        _new_metadata = {}
    return {"name": name, "scale": scale, "metadata": _new_metadata}


def _find_version(old_suffix, new_suffix):
    s_idx = old_suffix.find(new_suffix)
    if s_idx != -1:
        v_idx = s_idx + len(new_suffix)
        old_suffix, current_version = old_suffix[:v_idx], old_suffix[v_idx:]

        current_version = 0 if current_version == "" else int(current_version[1:-1])
        current_version += 1
        new_version = f"[{current_version}]"
        return old_suffix, new_version

    return f"{old_suffix}_{new_suffix}", ""


def create_layer_name(base, new_suffix):
    if base.find("_") == -1:
        return f"{base}_{new_suffix}"

    *base_without_suffix, old_suffix = base.split("_")
    base_without_suffix = "_".join(base_without_suffix)

    new_suffix, version = _find_version(old_suffix, new_suffix)
    return f"{base_without_suffix}_{new_suffix}{version}"


def return_value_if_widget(x):
    if isinstance(x, Widget):
        return x.value
    return x
