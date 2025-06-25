=== "PlantSeg Zoo"
    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")


    from napari_widgets_render import render_widget
    from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab
    from plantseg.viewer_napari.widgets.prediction import UNetPredictionMode

    tab = Segmentation_Tab()
    w = tab.prediction_widgets.widget_unet_prediction
    w.mode.value = UNetPredictionMode.PLANTSEG
    w.model_name.value = w.model_name.choices[1]
    w.model_id.hide()

    html = render_widget(w, skip_name=True, skip_doc=True)
    print(html)
    ```

=== "BioImage.IO Model Zoo"
    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")


    from napari_widgets_render import render_widget
    from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab
    from plantseg.viewer_napari.widgets.prediction import UNetPredictionMode

    tab = Segmentation_Tab()
    w = tab.prediction_widgets.widget_unet_prediction
    w.mode.value = UNetPredictionMode.BIOIMAGEIO

    html = render_widget(w, skip_name=True, skip_doc=True)
    print(html)
    ```
