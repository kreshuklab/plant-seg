from magicgui.widgets import Widget
from typing import Callable
import timeit
from concurrent.futures import Future

from napari.qt.threading import create_worker

from plantseg.napari.logging import napari_formatted_logging
from plantseg.image import PlantSegImage
import napari


def _return_value_if_widget(x):
    if isinstance(x, Widget):
        return x.value
    return x


def setup_layers_suggestions(out_name: str, widgets: list):
    """Update the widgets with the output of the task."""
    viewer = napari.current_viewer()

    if viewer is None:
        return None

    if widgets is None or not widgets:
        return None

    if out_name not in viewer.layers:
        return None

    out_layer = viewer.layers[out_name]
    for widget in widgets:
        widget.value = out_layer


def schedule_task(task: Callable, task_kwargs: dict, widget_to_update: list[Widget] | None = None):
    """Schedule a task to be executed in a separate thread and update the widgets with the result.

    Args:
        func (Callable): Function to be executed, the function should be a workflow task,
            and return a PlantSegImage or a tuple of PlantSegImage, or None.
        task_kwargs (dict): Keyword arguments for the function.
        widget_to_update (list[Widget], optional): Widgets to be updated with the result. Defaults to None.
    """

    if hasattr(task, '__plantseg_task__'):
        task_name = task.__plantseg_task__
    else:
        raise ValueError(f"Function {task.__name__} is not a PlantSeg task.")

    future = Future()
    timer_start = timeit.default_timer()

    def on_done(task_result: PlantSegImage | tuple[PlantSegImage, ...] | None):
        timer = timeit.default_timer() - timer_start
        napari_formatted_logging(f"{task_name} complete in {timer:.2f}s", thread='Task')

        if isinstance(task_result, PlantSegImage):
            future.set_result(task_result.to_napari_layer_tuple())
            setup_layers_suggestions(out_name=task_result.name, widgets=widget_to_update)

        elif isinstance(task_result, tuple) or isinstance(task_result, list):
            for ps_im in task_result:
                if not isinstance(ps_im, PlantSegImage):
                    raise ValueError(f"Task {task_name} returned an unexpected value {task_result}")

            layers_tuple = tuple([ps_im.to_napari_layer_tuple() for ps_im in task_result])
            future.set_result(layers_tuple)

            [setup_layers_suggestions(out_name=ps_im.name, widgets=widget_to_update) for ps_im in task_result]

        elif task_result is None:
            future.set_result(None)

        else:
            raise ValueError(f"Task {task_name} returned an unexpected value {task_result}")

    worker = create_worker(task, **task_kwargs)
    worker.returned.connect(on_done)
    worker.start()
    napari_formatted_logging(f"{task_name} started", thread='Task')
    return future
