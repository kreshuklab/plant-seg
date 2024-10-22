import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import widget_gaussian_smoothing

html = render_widget(widget_gaussian_smoothing)
print(html)
