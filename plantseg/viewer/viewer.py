import napari

from plantseg.viewer.containers import get_proofreading
from plantseg.viewer.containers import get_extra
from plantseg.viewer.containers import get_gasp_workflow, get_preprocessing_workflow, get_data_io
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.proofreading.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_data_io()
    main_w = viewer.window.add_dock_widget(main_container, name="Data")

    setup_proofreading_keybindings(viewer)

    for _containers, name in [
        (get_preprocessing_workflow(), "Data-Processing"),
        (get_gasp_workflow(), "Main"),
        (get_extra(), "Extra"),
        (get_proofreading(), "Proofreading"),
    ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    napari_formatted_logging("Plantseg is ready!", thread="run_viewer", level="info")
    napari.run()
