from magicgui.widgets import Container

from plantseg.napari.widget.dataprocessing import widget_rescaling, widget_gaussian_smoothing
from plantseg.napari.widget.dataprocessing import widget_cropping, widget_add_layers
from plantseg.napari.widget.io import open_file, export_stacks
from plantseg.napari.widget.predictions import widget_unet_predictions, widget_test_all_unet_predictions
from plantseg.napari.widget.predictions import widget_iterative_unet_predictions, widget_add_custom_model
from plantseg.napari.widget.segmentation import widget_dt_ws, widget_gasp
from plantseg.napari.widget.segmentation import widget_fix_over_under_segmentation_from_nuclei
from plantseg.napari.widget.segmentation import widget_multicut, widget_lifted_multicut
from plantseg.napari.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from plantseg.napari.widget.proofreading.proofreading import widget_clean_scribble
from plantseg.napari.widget.dataprocessing import widget_label_processing


def get_main():
    container = Container(widgets=[open_file,
                                   export_stacks,
                                   widget_split_and_merge_from_scribbles,
                                   widget_clean_scribble
                                   ],
                          labels=False)

    return container


def get_preprocessing_workflow():
    container = Container(widgets=[widget_gaussian_smoothing,
                                   widget_rescaling,
                                   widget_cropping,
                                   widget_add_layers,
                                   widget_label_processing],
                          labels=False)
    return container


def get_gasp_workflow():
    container = Container(widgets=[widget_unet_predictions,
                                   widget_dt_ws,
                                   widget_gasp],
                          labels=False)
    return container


def get_extra_seg():
    container = Container(widgets=[widget_multicut,
                                   widget_lifted_multicut,
                                   widget_fix_over_under_segmentation_from_nuclei],
                          labels=False)
    return container


def get_extra_pred():
    container = Container(widgets=[widget_test_all_unet_predictions,
                                   widget_iterative_unet_predictions,
                                   widget_add_custom_model],
                          labels=False)
    return container
