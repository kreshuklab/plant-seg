import napari

from plantseg.napari.containers import (
    get_data_io,
    get_preprocessing_tab,
    get_main_tab,
    get_extras_tab,
    get_proofreading_tab,
)
from plantseg.napari.logging import napari_formatted_logging

# from plantseg._viewer.widget.proofreading.proofreading import (
#     setup_proofreading_keybindings,
# )


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_data_io()
    main_w = viewer.window.add_dock_widget(main_container, name="Data")

    # setup_proofreading_keybindings(viewer)

    for _containers, name in [
        (get_preprocessing_tab(), "Preprocessing"),
        (get_main_tab(), "Main"),
        (get_extras_tab(), "Extra"),
        (get_proofreading_tab(), "Proofreading"),
    ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    napari_formatted_logging("Plantseg is ready!", thread="run_viewer", level="info")
    napari.run()
