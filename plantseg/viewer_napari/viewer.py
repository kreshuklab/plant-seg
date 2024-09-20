import napari

from plantseg.viewer_napari import log
from plantseg.viewer_napari.containers import (
    get_data_io_tab,
    get_extras_tab,
    get_main_tab,
    get_preprocessing_tab,
    get_proofreading_tab,
)
from plantseg.viewer_napari.widgets.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer(title='PlantSeg v2')
    setup_proofreading_keybindings(viewer)

    # Create and add tabs
    for _containers, name in [
        (get_data_io_tab(), "Data"),
        (get_preprocessing_tab(), "Preprocessing"),
        (get_main_tab(), "Main"),
        (get_proofreading_tab(), "Proofreading"),
        (get_extras_tab(), "Extra"),
    ]:
        viewer.window.add_dock_widget(_containers, name=name, tabify=True)

    # Show data tab by default
    viewer.window._dock_widgets['Data'].show()
    viewer.window._dock_widgets['Data'].raise_()

    log("Plantseg is ready!", thread="Run viewer", level="info")
    napari.run()
