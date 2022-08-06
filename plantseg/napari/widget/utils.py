from concurrent.futures import Future
from typing import Callable

from napari.qt.threading import thread_worker
import napari


def start_threading_process(func: Callable) -> Future:
    func = thread_worker(func)

    future = Future()

    def on_done(result):
        future.set_result(result)

    worker = func()
    worker.returned.connect(on_done)
    worker.start()
    return future


def choices_of_image_layers(viewer: napari.Viewer):
    return [layer.name for layer in viewer.layers if isinstance(layer,
                                                                napari.layers.image.image.Image)]


def choices_of_label_layers(viewer: napari.Viewer):
    return [layer.name for layer in viewer.layers if isinstance(layer,
                                                                napari.layers.labels.labels.Labels)]
