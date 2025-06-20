import napari
from qtpy import QtCore, QtWidgets

from plantseg.__version__ import __version__
from plantseg.utils import check_version
from plantseg.viewer_napari import log
from plantseg.viewer_napari.containers import (
    get_proofreading_tab,
)
from plantseg.viewer_napari.widgets.batch import Batch_Tab
from plantseg.viewer_napari.widgets.input import Input_Tab
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab
from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab
from plantseg.viewer_napari.widgets.proofreading import setup_proofreading_keybindings
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab


def scroll_wrap(w):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setWidget(w.native)
    scrollArea.setWidgetResizable(True)
    pol = QtWidgets.QSizePolicy()
    pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
    pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
    scrollArea.setSizePolicy(pol)
    # width of scroll area (outside)
    scrollArea.setMinimumWidth(550)

    return scrollArea


def run_viewer():
    viewer = napari.Viewer(title="PlantSeg v2")
    setup_proofreading_keybindings(viewer=viewer)
    input_tab = Input_Tab()
    output_tab = Output_Tab()
    preprocessing_tab = Preprocessing_Tab()
    segmentation_tab = Segmentation_Tab()
    postprocessing_tab = Postprocessing_Tab()
    batch_tab = Batch_Tab(output_tab)

    # Create and add tabs
    container_list = [
        (input_tab.get_container(), "Input"),
        (preprocessing_tab.get_container(), "Preprocessing"),
        (segmentation_tab.get_container(), "Segmentation"),
        (postprocessing_tab.get_container(), "Postprocessing"),
        (get_proofreading_tab(), "Proofreading"),
        (output_tab.get_container(), "Output"),
        (batch_tab.get_container(), "Batch"),
    ]
    for _containers, name in container_list:
        # width inside scroll area
        _containers.native.setFixedWidth(550)
        viewer.window.add_dock_widget(
            # breaks layer-name updates #439
            # scroll_wrap(_containers),
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
