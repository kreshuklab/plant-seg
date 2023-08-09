import webbrowser

from PyQt5.QtCore import Qt
from magicgui.widgets import MainWindow

from plantseg.viewer.widget.dataprocessing import widget_cropping, widget_add_layers
from plantseg.viewer.widget.dataprocessing import widget_label_processing
from plantseg.viewer.widget.dataprocessing import widget_rescaling, widget_gaussian_smoothing
from plantseg.viewer.widget.io import open_file, export_stacks
from plantseg.viewer.widget.predictions import widget_iterative_unet_predictions, widget_add_custom_model
from plantseg.viewer.widget.predictions import widget_unet_predictions, widget_test_all_unet_predictions
from plantseg.viewer.widget.proofreading.proofreading import widget_clean_scribble, widget_filter_segmentation
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from plantseg.viewer.widget.segmentation import widget_dt_ws, widget_agglomeration
from plantseg.viewer.widget.segmentation import widget_fix_over_under_segmentation_from_nuclei
from plantseg.viewer.widget.segmentation import widget_lifted_multicut
from plantseg.viewer.widget.segmentation import widget_simple_dt_ws
from plantseg.viewer.widget.train_dataset import widget_create_dataset, widget_print_dataset
from plantseg.viewer.widget.train_dataset import widget_add_stack, widget_delete_dataset


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
    container = MainWindow(widgets=[open_file,
                                    export_stacks,
                                    widget_split_and_merge_from_scribbles,
                                    widget_clean_scribble,
                                    widget_filter_segmentation,
                                    ],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/Napari-Main')
    return container


def get_preprocessing_workflow():
    container = MainWindow(widgets=[widget_gaussian_smoothing,
                                    widget_rescaling,
                                    widget_cropping,
                                    widget_add_layers,
                                    widget_label_processing],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/Data-Processing')
    return container


def get_gasp_workflow():
    container = MainWindow(widgets=[widget_unet_predictions,
                                    widget_simple_dt_ws,
                                    widget_agglomeration],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/UNet-GASP-Workflow')
    return container


def get_dataset_workflow():
    container = MainWindow(widgets=[widget_create_dataset,
                                    widget_print_dataset,
                                    widget_add_stack,
                                    widget_delete_dataset],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/Dataset-Managment')
    return container


def get_extra_seg():
    container = MainWindow(widgets=[widget_dt_ws,
                                    widget_lifted_multicut,
                                    widget_fix_over_under_segmentation_from_nuclei],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/Extra-Seg')
    return container


def get_extra_pred():
    container = MainWindow(widgets=[widget_test_all_unet_predictions,
                                    widget_iterative_unet_predictions,
                                    widget_add_custom_model],
                           labels=False)
    container = setup_menu(container, path='https://github.com/hci-unihd/plant-seg/wiki/Extra-Pred')
    return container
