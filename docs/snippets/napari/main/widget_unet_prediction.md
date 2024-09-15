
=== "PlantSeg Zoo"
    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")


    from napari_widgets_render import render_widget

    from plantseg.viewer_napari.widgets.prediction import widget_unet_prediction, UNetPredictionMode

    widget_unet_prediction.mode.value = UNetPredictionMode.PLANTSEG

    html = render_widget(widget_unet_prediction)
    print(html)
    ```

=== "BioImage.IO Model Zoo"
    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")


    from napari_widgets_render import render_widget

    from plantseg.viewer_napari.widgets.prediction import widget_unet_prediction, UNetPredictionMode

    widget_unet_prediction.mode.value = UNetPredictionMode.BIOIMAGEIO

    html = render_widget(widget_unet_prediction)
    print(html)
    ```
