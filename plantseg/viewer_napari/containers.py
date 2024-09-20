from magicgui.widgets import Container

from plantseg.viewer_napari.widgets import (
    widget_add_custom_model,
    widget_agglomeration,
    widget_clean_scribble,
    widget_dt_ws,
    widget_export_stacks,
    widget_filter_segmentation,
    widget_fix_over_under_segmentation_from_nuclei,
    widget_gaussian_smoothing,
    widget_infos,
    widget_open_file,
    widget_proofreading_initialisation,
    widget_redo,
    widget_remove_false_positives_by_foreground,
    widget_rescaling,
    widget_save_state,
    widget_show_info,
    widget_split_and_merge_from_scribbles,
    widget_undo,
    widget_unet_prediction,
)

STYLE_SLIDER = "font-size: 9pt;"


def get_data_io_tab():
    container = Container(
        widgets=[
            widget_open_file,
            widget_export_stacks,
            widget_show_info,
            widget_infos,
        ],
        labels=False,
    )
    return container


def get_preprocessing_tab():
    # widget_cropping.crop_z.native.setStyleSheet(STYLE_SLIDER)  # TODO: remove comment when widget_cropping is implemented
    container = Container(
        widgets=[
            widget_gaussian_smoothing,
            widget_rescaling,
        ],
        labels=False,
    )
    return container


def get_main_tab():
    container = Container(
        widgets=[
            widget_unet_prediction,
            widget_dt_ws,
            widget_agglomeration,
        ],
        labels=False,
    )
    return container


def get_extras_tab():
    container = Container(
        widgets=[
            widget_add_custom_model,
        ],
        labels=False,
    )
    return container


def get_proofreading_tab():
    widget_fix_over_under_segmentation_from_nuclei.threshold.native.setStyleSheet(STYLE_SLIDER)
    widget_fix_over_under_segmentation_from_nuclei.quantile.native.setStyleSheet(STYLE_SLIDER)
    container = Container(
        widgets=[
            widget_proofreading_initialisation,
            widget_split_and_merge_from_scribbles,
            widget_clean_scribble,
            widget_filter_segmentation,
            widget_undo,
            widget_redo,
            widget_save_state,
            widget_remove_false_positives_by_foreground,
            widget_fix_over_under_segmentation_from_nuclei,
        ],
        labels=False,
    )
    return container
