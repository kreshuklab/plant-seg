## Widget: Run Image Rescaling

=== "From factor"
    Using the `From factor` mode, the user can rescale the image by a multiplicate factor. 
    For example, if the image has a shape `(10, 10, 10)` and the user wants to rescale it by a factor of `(2, 2, 2)`, the new size will be `(20, 20, 20)`.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget
    from plantseg.viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "From factor"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "To layer voxel size"
    Using the `To layer voxel size` mode, the user can rescale the image to the voxel size of a specific layer.
    For example, if two images are loaded in the viewer, one with a voxel size of `(0.1, 0.1, 0.1)um` and the other with a voxel size of `(0.1, 0.05, 0.05)um`, the user can rescale the first image to the voxel size of the second image.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "To layer voxel size"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "To layer shape"
    Using the `To layer shape` mode, the user can rescale the image to the shape of a specific layer. For example, if two images are loaded in the viewer, one with a shape `(10, 10, 10)` and the other with a shape `(20, 20, 20)`, the user can rescale the first image to the shape of the second image.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "To layer shape"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "To model voxel size"
    Using the `To model voxel size` mode, the user can rescale the image to the voxel size of the model. 
    For example, if the model has been trained with data at voxel size of `(0.1, 0.1, 0.1)um`, the user can rescale the image to this voxel size.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "To model voxel size"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "To voxel size"
    Using the `To voxel size` mode, the user can rescale the image to a specific voxel size.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "To voxel size"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "To shape"
    Using the `To shape` mode, the user can rescale the image to a specific shape.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "To shape"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```

=== "Set voxel size"
    Using the `Set voxel size` mode, the user can set the voxel size of the image to a specific value. This only changes the metadata of the image and does not rescale the image.

    ```python exec="1" html="1"
    import sys

    sys.path.append("docs/snippets")
    from napari_widgets_render import render_widget

    sys.path.append("plantseg")
    from viewer.widget.dataprocessing import widget_rescaling

    widget_rescaling.mode.value = "Set voxel size"

    html = render_widget(widget_rescaling, skip_name=True)
    print(html)
    ```