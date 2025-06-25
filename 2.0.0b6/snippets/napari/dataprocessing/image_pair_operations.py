import sys

sys.path.append("docs/snippets")

from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import widget_image_pair_operations

html = render_widget(widget_image_pair_operations)
print(html)
