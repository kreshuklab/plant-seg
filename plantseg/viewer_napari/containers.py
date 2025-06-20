from magicgui.widgets import Container
from qtpy.QtGui import QFont

from plantseg.viewer_napari.widgets import (
    widget_clean_scribble,
    widget_docs,
    widget_filter_segmentation,
    widget_proofreading_initialisation,
    widget_redo,
    widget_save_state,
    widget_split_and_merge_from_scribbles,
    widget_undo,
)

STYLE_SLIDER = "font-size: 9pt;"
MONOSPACE_FONT = QFont("Courier New", 9)  # "Courier New" is a common monospaced font


def get_data_io_tab():
    container = Container(
        widgets=[
            widget_docs,
        ],
        labels=False,
    )
    return container


# def get_preprocessing_tab():
#     # widget_cropping.crop_z.native.setStyleSheet(STYLE_SLIDER)  # TODO: remove comment when widget_cropping is implemented
#     container = Container(
#         widgets=[
#             widget_cropping,
#             widget_rescaling,
#             widget_gaussian_smoothing,
#             widget_image_pair_operations,
#         ],
#         labels=False,
#     )
#     return container
#

#
# def get_postprocessing_tab():
#     container = Container(
#         widgets=[
#             widget_relabel,
#             widget_set_biggest_instance_to_zero,
#             widget_remove_false_positives_by_foreground,
#             widget_fix_over_under_segmentation_from_nuclei,
#         ],
#         labels=False,
#     )
#     return container
#


def get_proofreading_tab():
    # widget_fix_over_under_segmentation_from_nuclei.threshold.native.setStyleSheet(
    #     STYLE_SLIDER
    # )
    # widget_fix_over_under_segmentation_from_nuclei.quantile.native.setStyleSheet(
    #     STYLE_SLIDER
    # )
    container = Container(
        widgets=[
            widget_proofreading_initialisation,
            widget_split_and_merge_from_scribbles,
            widget_clean_scribble,
            widget_filter_segmentation,
            widget_undo,
            widget_redo,
            widget_save_state,
        ],
        labels=False,
    )
    return container
