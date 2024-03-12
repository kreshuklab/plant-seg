import webbrowser

from PyQt5.QtCore import Qt
from magicgui.widgets import MainWindow

from plantseg.viewer.widget.dataprocessing import widget_cropping, widget_add_layers
from plantseg.viewer.widget.dataprocessing import widget_label_processing
from plantseg.viewer.widget.dataprocessing import widget_rescaling, widget_gaussian_smoothing
from plantseg.viewer.widget.io import open_file_widget, export_stacks
from plantseg.viewer.widget.predictions import widget_iterative_unet_predictions, widget_add_custom_model
from plantseg.viewer.widget.predictions import widget_unet_predictions, widget_test_all_unet_predictions
from plantseg.viewer.widget.proofreading.proofreading import widget_clean_scribble, widget_filter_segmentation
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from plantseg.viewer.widget.segmentation import widget_dt_ws, widget_agglomeration
from plantseg.viewer.widget.segmentation import widget_fix_over_under_segmentation_from_nuclei
from plantseg.viewer.widget.segmentation import widget_fix_false_positive_from_foreground_pmap
from plantseg.viewer.widget.segmentation import widget_lifted_multicut
from plantseg.viewer.widget.segmentation import widget_simple_dt_ws


def setup_menu(container, path=None):
    def _callback():
        if path is not None:
            webbrowser.open(path)

    container.create_menu_item(menu_name='Help',
                               item_name='Open Documentation',
                               callback=_callback)

    container._widget._layout.setAlignment(Qt.AlignTop)
    return container


def get_main():
    container = MainWindow(widgets=[open_file_widget,
                                    export_stacks,
                                    widget_split_and_merge_from_scribbles,
                                    widget_clean_scribble,
                                    widget_filter_segmentation,
                                    ],
                           labels=False)
    container = setup_menu(container, path='https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/index.html')
    return container


def get_preprocessing_workflow():
    container = MainWindow(widgets=[widget_gaussian_smoothing,
                                    widget_rescaling,
                                    widget_cropping,
                                    widget_add_layers,
                                    widget_label_processing],
                           labels=False)
    container = setup_menu(container, path='https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/data_processing.html')
    return container


def get_gasp_workflow():
    container = MainWindow(widgets=[widget_unet_predictions,
                                    widget_simple_dt_ws,
                                    widget_agglomeration],
                           labels=False)
    container = setup_menu(container, path='https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/unet_gasp_workflow.html')
    return container


def get_extra_seg():
    container = MainWindow(widgets=[widget_dt_ws,
                                    widget_lifted_multicut,
                                    widget_fix_over_under_segmentation_from_nuclei,
                                    widget_fix_false_positive_from_foreground_pmap],
                           labels=False)
    container = setup_menu(container, path='https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/extra_seg.html')
    return container


def get_extra_pred():
    container = MainWindow(widgets=[widget_test_all_unet_predictions,
                                    widget_iterative_unet_predictions,
                                    widget_add_custom_model],
                           labels=False)
    container = setup_menu(container, path='https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/extra_pred.html')
    return container
