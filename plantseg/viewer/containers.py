import webbrowser

from PyQt5.QtCore import Qt
from magicgui.widgets import MainWindow, Label

from plantseg.viewer.widget.dataprocessing import widget_cropping, widget_merge_layers
from plantseg.viewer.widget.dataprocessing import widget_label_processing
from plantseg.viewer.widget.dataprocessing import widget_rescaling, widget_gaussian_smoothing
from plantseg.viewer.widget.extra_widgets import widget_add_custom_model
from plantseg.viewer.widget.io import widget_open_file, widget_export_stacks
from plantseg.viewer.widget.predictions import widget_iterative_unet_predictions
from plantseg.viewer.widget.predictions import widget_unet_predictions, widget_test_all_unet_predictions
from plantseg.viewer.widget.proofreading.proofreading import widget_clean_scribble, widget_filter_segmentation
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from plantseg.viewer.widget.segmentation import widget_dt_ws, widget_agglomeration
from plantseg.viewer.widget.segmentation import widget_fix_over_under_segmentation_from_nuclei
from plantseg.viewer.widget.segmentation import widget_fix_false_positive_from_foreground_pmap
from plantseg.viewer.widget.extra_widgets import widget_extra_manager
from plantseg.viewer.widget.segmentation import widget_lifted_multicut

STYLE_SLIDER = "font-size: 9pt;"


def setup_menu(container, path=None):
    def _callback():
        if path is not None:
            webbrowser.open(path)

    container.create_menu_item(menu_name='Help', item_name='Open Documentation', callback=_callback)

    container._widget._layout.setAlignment(Qt.AlignTop)
    return container


def get_data_io():
    container = MainWindow(widgets=[widget_open_file, widget_export_stacks], labels=False)
    container = setup_menu(
        container, path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/data_io.html'
    )
    return container


def get_preprocessing_workflow():
    widget_cropping.crop_z.native.setStyleSheet(STYLE_SLIDER)
    container = MainWindow(
        widgets=[
            widget_gaussian_smoothing,
            widget_rescaling,
            widget_cropping,
            widget_merge_layers,
            widget_label_processing,
        ],
        labels=False,
    )
    container = setup_menu(
        container,
        path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/data_processing.html',
    )
    return container


def get_gasp_workflow():
    container = MainWindow(widgets=[widget_unet_predictions, widget_dt_ws, widget_agglomeration], labels=False)
    container = setup_menu(
        container,
        path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/unet_gasp_workflow.html',
    )
    return container


def get_extra():
    widget_fix_over_under_segmentation_from_nuclei.threshold.native.setStyleSheet(STYLE_SLIDER)
    widget_fix_over_under_segmentation_from_nuclei.quantile.native.setStyleSheet(STYLE_SLIDER)
    container = MainWindow(
        widgets=[
            widget_extra_manager,
            Label(),
            widget_lifted_multicut,
            widget_add_custom_model,
            widget_test_all_unet_predictions,
            widget_iterative_unet_predictions,
            widget_fix_over_under_segmentation_from_nuclei,
        ],
        labels=False,
    )
    container = setup_menu(
        container, path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/extra.html'
    )
    return container


def get_proofreading():
    container = MainWindow(
        widgets=[
            widget_split_and_merge_from_scribbles,
            widget_clean_scribble,
            widget_filter_segmentation,
            widget_fix_false_positive_from_foreground_pmap,
        ],
        labels=False,
    )
    container = setup_menu(
        container, path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/proofreading.html'
    )
    return container
