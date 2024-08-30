import timeit
from concurrent.futures import Future
from typing import Callable

import napari
from magicgui.widgets import Widget
from napari.qt.threading import create_worker

from plantseg.core.image import PlantSegImage
from plantseg.viewer_napari import log


def _return_value_if_widget(x):
    if isinstance(x, Widget):
        return x.value
    return x


def setup_layers_suggestions(out_name: str, widgets: list[Widget] | None):
    """Update the widgets with the output of the task.

    Args:
        out_name (str): The name of the output layer.
        widgets (list[Widget] | None): List of widgets to be updated, if any.
    """
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


def schedule_task(task: Callable, task_kwargs: dict, widgets_to_update: list[Widget] | None = None) -> Future:
    """Schedule a task to be executed in a separate thread and update the widgets with the result.

    Args:
        task (Callable): Function to be executed, the function should be a workflow task,
            and return a PlantSegImage or a tuple/list of PlantSegImage, or None.
        task_kwargs (dict): Keyword arguments for the function.
        widgets_to_update (list[Widget] | None, optional): Widgets to be updated with the result. Defaults to None.

    Returns:
        Future: A Future object representing the asynchronous execution of the task.
    """

    if hasattr(task, '__plantseg_task__'):
        task_name = task.__plantseg_task__
    else:
        raise ValueError(f"Function {task.__name__} is not a PlantSeg task.")

    future = Future()
    timer_start = timeit.default_timer()

    def on_done(task_result: PlantSegImage | list[PlantSegImage] | None):
        timer = timeit.default_timer() - timer_start
        log(f"{task_name} complete in {timer:.2f}s", thread='Task')

        if isinstance(task_result, PlantSegImage):
            future.set_result(task_result.to_napari_layer_tuple())
            setup_layers_suggestions(out_name=task_result.name, widgets=widgets_to_update)

        elif isinstance(task_result, (tuple, list)):
            for ps_im in task_result:
                if not isinstance(ps_im, PlantSegImage):
                    raise ValueError(f"Task {task_name} returned an unexpected value {task_result}")

            future.set_result([ps_im.to_napari_layer_tuple() for ps_im in task_result])
            setup_layers_suggestions(out_name=task_result[-1].name, widgets=widgets_to_update)

        elif task_result is None:
            future.set_result(None)

        else:
            raise ValueError(f"Task {task_name} returned an unexpected value {task_result}")

    worker = create_worker(task, **task_kwargs)
    worker.returned.connect(on_done)
    worker.start()
    log(f"{task_name} started", thread='Task')
    return future
