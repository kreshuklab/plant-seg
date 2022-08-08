from concurrent.futures import Future
from typing import Callable, Tuple

from napari.qt.threading import thread_worker
import napari
from functools import partial
from plantseg.napari.dag_manager import dag


def identity(x):
    return x


def start_threading_process(func: Callable,
                            func_kwargs: dict,
                            out_name: str,
                            input_keys: Tuple[str, ...],
                            layer_kwarg: dict,
                            layer_type='image',
                            skip_dag=False) -> Future:
    thread_func = thread_worker(partial(func, **func_kwargs))
    future = Future()

    def on_done(result):
        _func = func if not skip_dag else identity
        dag.add_step(_func, input_keys=input_keys, output_key=out_name)
        result = result, layer_kwarg, layer_type
        future.set_result(result)

    worker = thread_func()
    worker.returned.connect(on_done)
    worker.start()
    return future


def layer_properties(name, scale):
    return {'name': name, 'scale': scale}


def choices_of_image_layers(viewer: napari.Viewer):
    return [layer.name for layer in viewer.layers if isinstance(layer,
                                                                napari.layers.image.image.Image)]


def choices_of_label_layers(viewer: napari.Viewer):
    return [layer.name for layer in viewer.layers if isinstance(layer,
                                                                napari.layers.labels.labels.Labels)]
