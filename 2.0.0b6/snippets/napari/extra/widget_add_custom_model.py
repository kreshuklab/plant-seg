import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import widget_add_custom_model

html = render_widget(widget_add_custom_model)
print(html)
