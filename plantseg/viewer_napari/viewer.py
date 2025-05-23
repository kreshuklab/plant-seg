import napari
from qtpy import QtCore, QtWidgets

from plantseg.__version__ import __version__
from plantseg.utils import check_version
from plantseg.viewer_napari import log
from plantseg.viewer_napari.containers import (
    get_data_io_tab,
    get_postprocessing_tab,
    get_preprocessing_tab,
    get_proofreading_tab,
    get_segmentation_tab,
)
from plantseg.viewer_napari.widgets.dataprocessing import on_layer_rename_dataprocessing
from plantseg.viewer_napari.widgets.io import on_layer_rename_io
from plantseg.viewer_napari.widgets.prediction import on_layer_rename_prediction
from plantseg.viewer_napari.widgets.proofreading import setup_proofreading_keybindings
from plantseg.viewer_napari.widgets.segmentation import on_layer_rename_segmentation


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
    setup_proofreading_keybindings(viewer=viewer)

    # Create and add tabs
    for _containers, name in [
        (get_data_io_tab(), "Input/Output"),
        (get_preprocessing_tab(), "Preprocessing"),
        (get_segmentation_tab(), "Segmentation"),
        (get_postprocessing_tab(), "Postprocessing"),
        (get_proofreading_tab(), "Proofreading"),
    ]:
        _containers.native.setMinimumWidth(550)
        viewer.window.add_dock_widget(
            # breaks layer-name updates #439
            # scroll_wrap(_containers),
            _containers,
            name=name,
            tabify=True,
        )
        # allow content to float to top of dock
        _containers.native.layout().addStretch()

    # update layer drop-down menus on layer selection
    viewer.layers.selection.events.active.connect(on_layer_rename_prediction())
    viewer.layers.selection.events.active.connect(on_layer_rename_io())
    viewer.layers.selection.events.active.connect(on_layer_rename_dataprocessing())
    viewer.layers.selection.events.active.connect(on_layer_rename_segmentation())

    # Show data tab by default
    viewer.window._dock_widgets["Input/Output"].show()
    viewer.window._dock_widgets["Input/Output"].raise_()
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

    log("Plantseg is ready!", thread="Run viewer", level="info")
    napari.run()
