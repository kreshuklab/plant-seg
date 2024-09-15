import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import widget_agglomeration

html = render_widget(widget_agglomeration)
print(html)
