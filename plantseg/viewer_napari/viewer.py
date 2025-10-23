import napari
from qtpy import QtCore, QtWidgets

from plantseg.__version__ import __version__
from plantseg.utils import check_version
from plantseg.viewer_napari.containers import (
    get_proofreading_tab,
)
from plantseg.viewer_napari.updater import update
from plantseg.viewer_napari.widgets import proofreading
from plantseg.viewer_napari.widgets.input import Input_Tab
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.postprocessing import Postprocessing_Tab
from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab
from plantseg.viewer_napari.widgets.training import Training_Tab
from plantseg.viewer_napari.widgets.utils import decrease_font_size, increase_font_size


def scroll_wrap(w):
    scrollArea = QtWidgets.QScrollArea()
    scrollArea.setWidget(w.native)
    scrollArea.setWidgetResizable(True)
    pol = QtWidgets.QSizePolicy()
    pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Minimum)
    pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
    scrollArea.setSizePolicy(pol)
    return scrollArea


def run_viewer():
    viewer = napari.Viewer(title="PlantSeg v2")
    viewer.window.file_menu.menuAction().setVisible(False)
    viewer.window.layers_menu.menuAction().setVisible(False)

    # By hiding the menu, also closing shortcuts are removed
    @viewer.bind_key("Ctrl+w")
    def _close_window(viewer):
        viewer.window._qt_window.close(False, True)

    @viewer.bind_key("Ctrl+q")
    def _close_app(viewer):
        viewer.window._qt_window.close(True, True)

    input_tab = Input_Tab()
    output_tab = Output_Tab()
    preprocessing_tab = Preprocessing_Tab()
    segmentation_tab = Segmentation_Tab()
    postprocessing_tab = Postprocessing_Tab()
    training_tab = Training_Tab(segmentation_tab.prediction_widgets)

    # Create and add tabs
    container_list = [
        (input_tab.get_container(), "Input"),
        (preprocessing_tab.get_container(), "Preprocessing"),
        (segmentation_tab.get_container(), "Segmentation"),
        (postprocessing_tab.get_container(), "Postprocessing"),
        (get_proofreading_tab(), "Proofreading"),
        (output_tab.get_container(), "Output"),
        (training_tab.get_container(), "Train"),
    ]
    for _containers, name in container_list:
        # _containers.native.setMaximumHeight(600)
        # allow content to float to top of dock
        _containers.native.layout().addStretch()
        _containers.native.setMinimumWidth(400)
        _containers = scroll_wrap(_containers)
        _containers.setMinimumWidth(350)
        viewer.window.add_dock_widget(
            _containers,
            name=name,
            tabify=True,
        )

    # Drop-down update for new layers
    viewer.layers.events.inserted.connect(preprocessing_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(segmentation_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(postprocessing_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(training_tab.update_layer_selection)
    viewer.layers.events.inserted.connect(proofreading.update_layer_selection)

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
    viewer.layers.selection.events.active.connect(training_tab.update_layer_selection)
    viewer.layers.selection.events.active.connect(proofreading.update_layer_selection)

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

    viewer.window.plugins_menu.addAction("Update PlantSeg", update)
    viewer.window.view_menu.addAction(
        "Increase font size", increase_font_size, "Ctrl+i"
    )
    viewer.window.view_menu.addAction(
        "Decrease font size", decrease_font_size, "Ctrl+o"
    )

    napari.run()
