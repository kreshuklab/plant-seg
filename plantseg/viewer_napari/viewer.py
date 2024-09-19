import napari

from plantseg.viewer_napari import log
from plantseg.viewer_napari.containers import (
    get_data_io,
    get_extras_tab,
    get_main_tab,
    get_preprocessing_tab,
    get_proofreading_tab,
    get_training_tab,
)
from plantseg.viewer_napari.widgets.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_data_io()
    main_w = viewer.window.add_dock_widget(main_container, name="Data")

    setup_proofreading_keybindings(viewer)

    for _containers, name in [
        (get_preprocessing_tab(), "Preprocessing"),
        (get_main_tab(), "Main"),
        (get_extras_tab(), "Extra"),
        (get_proofreading_tab(), "Proofreading"),
        (get_training_tab(), "Training"),
    ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    log("Plantseg is ready!", thread="Run viewer", level="info")
    napari.run()
