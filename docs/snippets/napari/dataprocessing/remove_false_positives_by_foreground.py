import sys

sys.path.append("docs/snippets")


from napari_widgets_render import render_widget

from plantseg.viewer_napari.widgets import widget_remove_false_positives_by_foreground

html = render_widget(widget_remove_false_positives_by_foreground)
print(html)
