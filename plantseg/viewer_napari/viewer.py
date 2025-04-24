import napari
from qtpy import QtWidgets

from plantseg.viewer_napari import log
from plantseg.viewer_napari.containers import (
    get_data_io_tab,
    get_postprocessing_tab,
    get_preprocessing_tab,
    get_proofreading_tab,
    get_segmentation_tab,
)
from plantseg.viewer_napari.widgets.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer(title="PlantSeg v2")
    setup_proofreading_keybindings(viewer=viewer)

    # Create and add tabs
    for _containers, name in [
        (get_data_io_tab(), "Input/Output"),
        (get_preprocessing_tab(), "Preprocessing"),
        (get_segmentation_tab(), "Segmentation"),
        (get_postprocessing_tab(), "Postprocessing"),
        (get_proofreading_tab(), "Proofreading"),
    ]:
        this_widget = viewer.window.add_dock_widget(_containers, name=name, tabify=True)
        this_widget.setFixedWidth(666)

    # Show data tab by default
    viewer.window._dock_widgets["Input/Output"].show()
    viewer.window._dock_widgets["Input/Output"].raise_()
    # viewer.window._qt_viewer.set_welcome_visible(False)
    welcome_widget = viewer.window._qt_viewer._welcome_widget

    for i, child in enumerate(welcome_widget.findChildren(QtWidgets.QWidget)):
        if isinstance(child, QtWidgets.QLabel):
            if i == 3:
                child.setText(
                    "Welcome to PlantSeg!\n\nTo load an image use the menu on the right"
                )
            else:
                child.setText("")

    log("Plantseg is ready!", thread="Run viewer", level="info")
    napari.run()
