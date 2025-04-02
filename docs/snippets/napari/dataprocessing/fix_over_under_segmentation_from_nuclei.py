import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import (
    widget_fix_over_under_segmentation_from_nuclei,
)

html = render_widget(widget_fix_over_under_segmentation_from_nuclei)
print(html)
