from magicgui.widgets import Container

from plantseg.viewer_napari.widgets import (
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


def get_proofreading_tab():
    container = Container(
        widgets=[
            widget_label_split_merge,
            widget_proofreading_initialisation,
            widget_split_and_merge_from_scribbles,
            widget_clean_scribble,
            widget_label_extraction,
            widget_filter_segmentation,
            widget_undo,
            widget_redo,
            widget_save_state,
        ],
        labels=False,
    )
    return container
