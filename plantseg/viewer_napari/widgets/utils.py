import timeit
from dataclasses import dataclass
from typing import Callable

import napari
from magicgui.widgets import ProgressBar, Widget
from napari.qt.threading import create_worker
from psygnal import evented
from psygnal.qt import start_emitting_from_queue

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


def add_ps_image_to_viewer(layer: PlantSegImage, replace: bool = False) -> None:
    """Add a PlantSegImage to the viewer.

    Args:
        layer (PlantSegImage): PlantSegImage to be added to the viewer.
        replace (bool): If True, the layer with the same name will be replaced. Defaults to False.
    """
    viewer = napari.current_viewer()
    data, meta, layer_type = layer.to_napari_layer_tuple()

    name = meta.get("name", None)
    if replace and name in viewer.layers:
        viewer.layers.remove(name)

    if layer_type == "image":
        viewer.add_image(data, **meta)
    elif layer_type == "labels":
        viewer.add_labels(data, **meta)


@evented
@dataclass()
class PBar_Tracker:
    """Keeps track of a progress bar

    Attributes:
        progress: must change every step to trigger update
        total: number of steps the progress bar should have.
            Set to 0 to have an indetermined progress bar.
    """

    progress: int = 0
    total: int = 0


def update_progressbar(pbar: ProgressBar, tracker: PBar_Tracker):
    """Increment progress bar. Creates a callback to be attached to a tracker.
    Args:
        pbar: progress bar to update
        tracker: Tracker instance of this progress bar.

    Returns:
        Update callback function
    """
    return lambda: (setattr(pbar, "max", tracker.total), pbar.increment(1))


def schedule_task(
    task: Callable, task_kwargs: dict, widgets_to_update: list[Widget] | None = None
) -> None:
    """Schedule a task to be executed in a separate thread and update the widgets with the result.

    Args:
        task (Callable): Function to be executed, the function should be a workflow task,
            and return a PlantSegImage or a tuple/list of PlantSegImage, or None.
        task_kwargs (dict): Keyword arguments for the function.
        widgets_to_update (list[Widget] | None, optional): Widgets to be updated with the result. Defaults to None.
    """

    if hasattr(task, "__plantseg_task__"):
        task_name = task.__plantseg_task__
    else:
        raise ValueError(f"Function {task.__name__} is not a PlantSeg task.")

    timer_start = timeit.default_timer()

    def on_done(task_result: PlantSegImage | list[PlantSegImage] | None):
        timer = timeit.default_timer() - timer_start
        log(f"{task_name} complete in {timer:.2f}s", thread="Task")

        if isinstance(task_result, PlantSegImage):
            add_ps_image_to_viewer(task_result, replace=True)
            setup_layers_suggestions(
                out_name=task_result.name, widgets=widgets_to_update
            )

        elif isinstance(task_result, (tuple, list)):
            for ps_im in task_result:
                if not isinstance(ps_im, PlantSegImage):
                    raise ValueError(
                        f"Task {task_name} returned an unexpected value {task_result}"
                    )

            for ps_im in task_result:
                add_ps_image_to_viewer(ps_im, replace=True)
            setup_layers_suggestions(
                out_name=task_result[-1].name, widgets=widgets_to_update
            )

        elif task_result is None:
            return None

        else:
            raise ValueError(
                f"Task {task_name} returned an unexpected value {task_result}"
            )

    # Setup progress bar
    pbar = None
    if "_pbar" in task_kwargs:
        tracker = PBar_Tracker()
        pbar = task_kwargs.pop("_pbar")
        pbar.max = tracker.total
        pbar.visible = True
        tracker.events.progress.connect(
            update_progressbar(pbar, tracker), thread="main"
        )
        task_kwargs["_tracker"] = tracker
        start_emitting_from_queue()
    if hide_list := task_kwargs.pop("_to_hide", None):
        for to_hide in hide_list:
            to_hide.visible = False

    # _progress displays spinner for all tasks, not necessary for magicgui progress bar.
    worker = create_worker(task, **task_kwargs, _progress=True)
    worker.returned.connect(on_done)

    # Hide progress bar after task
    if pbar is not None:
        worker.returned.connect(lambda _: setattr(pbar, "visible", False))
    if hide_list is not None:
        worker.returned.connect(
            lambda _: [setattr(w, "visible", True) for w in hide_list]
        )
    worker.start()
    log(f"{task_name} started", thread="Task")
    return None
