# from plantseg.viewer_napari.widgets.dataprocessing import (
#     widget_cropping,
#     widget_fix_over_under_segmentation_from_nuclei,
#     widget_gaussian_smoothing,
#     widget_image_pair_operations,
#     widget_relabel,
#     widget_remove_false_positives_by_foreground,
#     widget_rescaling,
#     widget_set_biggest_instance_to_zero,
# )
from plantseg.viewer_napari.widgets.docs import widget_docs

# from plantseg.viewer_napari.widgets.io import (
#     widget_edit_worflow,
#     widget_export_headless_workflow,
#     widget_export_image,
#     widget_infos,
#     widget_open_file,
#     widget_set_voxel_size,
#     widget_show_info,
# )
# from plantseg.viewer_napari.widgets.prediction_old import (
#     widget_add_custom_model,
#     widget_add_custom_model_toggl,
#     widget_unet_prediction,
# )
from plantseg.viewer_napari.widgets.proofreading import (
    widget_add_label_to_corrected,
    widget_clean_scribble,
    widget_filter_segmentation,
    widget_label_extraction,
    widget_label_split_merge,
    widget_proofreading_initialisation,
    widget_redo,
    widget_save_state,
    widget_split_and_merge_from_scribbles,
    widget_undo,
)

# from plantseg.viewer_napari.widgets.segmentation_old import (
#     widget_agglomeration,
#     widget_dt_ws,
# )

# __all__ = [
#     # Home
#     "widget_docs",
#     # Data processing
#     "widget_gaussian_smoothing",
#     "widget_rescaling",
#     "widget_cropping",
#     "widget_image_pair_operations",
#     # IO
#     # Main - Prediction
#     # Main - Segmentation
#     # Extra
#     "widget_relabel",
#     "widget_set_biggest_instance_to_zero",
#     # Proofreading
#     "widget_proofreading_initialisation",
#     "widget_split_and_merge_from_scribbles",
#     "widget_clean_scribble",
#     "widget_filter_segmentation",
#     "widget_undo",
#     "widget_redo",
#     "widget_save_state",
#     "widget_add_label_to_corrected",  # XXX: Not used in container
#     "widget_remove_false_positives_by_foreground",
#     "widget_fix_over_under_segmentation_from_nuclei",
# ]
