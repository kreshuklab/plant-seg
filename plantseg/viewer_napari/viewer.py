import napari
from qtpy import QtCore, QtWidgets

from plantseg.__version__ import __version__
from plantseg.utils import check_version
from plantseg.viewer_napari.containers import (
    get_proofreading_tab,
)
from plantseg.viewer_napari.widgets.input import Input_Tab
from plantseg.viewer_napari.widgets.misc import Misc_Tab
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab
from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab


def run_viewer():
    viewer = napari.Viewer(title="PlantSeg v2")
    # setup_proofreading_keybindings(viewer=viewer)
    input_tab = Input_Tab()
    output_tab = Output_Tab()
    preprocessing_tab = Preprocessing_Tab()
    segmentation_tab = Segmentation_Tab()
    postprocessing_tab = Postprocessing_Tab()
    misc_tab = Misc_Tab(output_tab, segmentation_tab.prediction_widgets)

    # Create and add tabs
    container_list = [
        (input_tab.get_container(), "Input"),
        (preprocessing_tab.get_container(), "Preprocessing"),
        (segmentation_tab.get_container(), "Segmentation"),
        (postprocessing_tab.get_container(), "Postprocessing"),
        (get_proofreading_tab(), "Proofreading"),
        (output_tab.get_container(), "Output"),
        (misc_tab.get_container(), "Train"),
    ]
    for _containers, name in container_list:
        _containers.native.setFixedWidth(550)
        viewer.window.add_dock_widget(
            _containers,
            name=name,
            tabify=True,
        )
        # allow content to float to top of dock
        _containers.native.layout().addStretch()

    # Drop-down update for new layers
    viewer.layers.events.inserted.connect(preprocessing_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(segmentation_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(postprocessing_tab.update_layer_selection)

    # Drop-down update for renaming of layers
    viewer.layers.selection.events.active.connect(input_tab.update_layer_selection)
    viewer.layers.selection.events.active.connect(
        preprocessing_tab.update_layer_selection
    )
    viewer.layers.selection.events.active.connect(
        segmentation_tab.update_layer_selection
    )
    viewer.layers.selection.events.active.connect(
        postprocessing_tab.update_layer_selection
    )

    # Show data tab by default
    viewer.window._dock_widgets["Input"].show()
    viewer.window._dock_widgets["Input"].raise_()
    # viewer.window._qt_viewer.set_welcome_visible(False)
    welcome_widget = viewer.window._qt_viewer._welcome_widget

    v_short, v_features = check_version(current_version=__version__, silent=True)

    for i, child in enumerate(welcome_widget.findChildren(QtWidgets.QWidget)):
        if isinstance(child, QtWidgets.QLabel):
            if i == 3:
                child.setText(
                    "Welcome to PlantSeg!\n\nTo load an image use the menu on the right\n\n"
                    + v_short
                    + "\n\n"
                    + v_features
                )
            else:
                child.setText("")
            child.setAlignment(QtCore.Qt.AlignLeft)

    # log("Plantseg is ready!", thread="Run viewer", level="info")
    napari.run()
