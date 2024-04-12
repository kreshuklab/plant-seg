"""Widget input validation"""

from napari.layers import Image
from magicgui.widgets import Widget


def widgets_inactive(*widgets, active):
    """Toggle visibility of widgets."""
    for widget in widgets:
        widget.visible = active


def widgets_valid(*widgets, valid):
    """Toggle background warning color of widgets."""
    for widget in widgets:
        widget.native.setStyleSheet("" if valid else "background-color: lightcoral")


def get_image_volume_from_layer(image):
    """Used for widget parameter validation in change-handlers."""
    image = image.data[0] if image.multiscale else image.data
    if not all(hasattr(image, attr) for attr in ("shape", "ndim", "__getitem__")):
        from numpy import asanyarray

        image = asanyarray(image)
    return image


def _on_prediction_input_image_change(widget: Widget, image: Image):
    shape = get_image_volume_from_layer(image).shape
    ndim = len(shape)
    widget.image.tooltip = f"Shape: {shape}"

    size_z = widget.patch_size[0]
    halo_z = widget.patch_halo[0]
    if ndim == 2 or (ndim == 3 and shape[0] == 1):  # 2D image imported by Napari thus no Z, or by PlantSeg widget
        size_z.value = 0
        halo_z.value = 0
        widgets_inactive(size_z, halo_z, active=False)
    elif ndim == 3 and shape[0] > 1:  # 3D
        size_z.value = min(64, shape[0])  # TODO: fetch model default
        halo_z.value = 8
        widgets_inactive(size_z, halo_z, active=True)
    else:
        raise ValueError(f"Unsupported number of dimensions: {ndim}")
