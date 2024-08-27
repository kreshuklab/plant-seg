from plantseg.viewer_napari.widgets.dataprocessing import widget_gaussian_smoothing, widget_rescaling
from plantseg.viewer_napari.widgets.io import widget_export_stacks, widget_infos, widget_open_file, widget_show_info
from plantseg.viewer_napari.widgets.predictions import widget_add_custom_model, widget_unet_predictions
from plantseg.viewer_napari.widgets.proofreading import (
    widget_add_label_to_corrected,
    widget_clean_scribble,
    widget_filter_segmentation,
    widget_split_and_merge_from_scribbles,
)
from plantseg.viewer_napari.widgets.segmentation import widget_agglomeration, widget_dt_ws, widget_lifted_multicut

__all__ = [
    # Data processing
    "widget_gaussian_smoothing",
    "widget_rescaling",
    # IO
    "widget_open_file",
    "widget_export_stacks",
    "widget_show_info",
    "widget_infos",
    # Main - Prediction
    "widget_unet_predictions",
    # Main - Segmentation
    "widget_dt_ws",
    "widget_agglomeration",
    # Extra - Segmentation
    "widget_lifted_multicut",
    "widget_add_custom_model",
    # Proofreading
    "widget_split_and_merge_from_scribbles",
    "widget_clean_scribble",
    "widget_filter_segmentation",
    "widget_add_label_to_corrected",  # XXX: Not used in container
]
