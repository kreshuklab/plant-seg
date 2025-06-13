from magicgui import magicgui
from napari.types import LayerDataTuple

from plantseg.core.image import PlantSegImage
from plantseg.viewer_napari.widgets.dataprocessing import (
    widget_fix_over_under_segmentation_from_nuclei,
)
from tests.tasks.dataprocessing.test_advanced_dataprocessing_tasks import (
    complex_test_PlantSegImages,  # noqa: F401
)


@magicgui
def widget_add_image(image: PlantSegImage) -> LayerDataTuple:
    """Add a plantseg.core.image.PlantSegImage to napari viewer as a napari.layers.Layer."""
    return image.to_napari_layer_tuple()


def test_widget_fix_over_under_segmentation_from_nuclei(
    qtbot, make_napari_viewer_proxy, complex_test_PlantSegImages
):
    """
    Test the widget_fix_over_under_segmentation_from_nuclei function in a napari viewer environment.

    Args:
        make_napari_viewer_proxy (function): A factory function to create a mock napari viewer instance.
        complex_test_PlantSegImages (tuple): Mock data containing:
            - cell_seg (PlantSegImage): PlantSegImage object for cell segmentation.
            - nuclei_seg (PlantSegImage): PlantSegImage object for nuclei segmentation.
            - boundary_pmap (PlantSegImage | None): PlantSegImage object for boundary probability map, or None.

    Tests:
        - Adds test PlantSegImage objects to the napari viewer.
        - Ensures the widget correctly processes the layers and outputs a corrected segmentation.
        - Verifies the properties of the corrected segmentation layer, including its name, shape, and data type.
    """
    # Create a mock napari viewer
    viewer = make_napari_viewer_proxy()

    # Extract test PlantSegImage objects
    cell_seg, nuclei_seg, boundary_pmap = complex_test_PlantSegImages
    widget_add_image(cell_seg)
    widget_add_image(nuclei_seg)
    if boundary_pmap is not None:
        widget_add_image(boundary_pmap)

    # Run the widget for correcting segmentation
    count_layers = len(viewer.layers)
    widget_fix_over_under_segmentation_from_nuclei(
        segmentation_cells=viewer.layers[cell_seg.name],
        segmentation_nuclei=viewer.layers[nuclei_seg.name],
        boundary_pmaps=(
            viewer.layers[boundary_pmap.name] if boundary_pmap is not None else None
        ),
        threshold=(30, 60),  # Threshold range as percentages (30% merge, 60% split)
        quantile=(10, 90),  # Quantile range as percentages (10%-90%)
    )
    qtbot.waitUntil(lambda: count_layers < len(viewer.layers), timeout=20000)

    corrected_layer = viewer.layers[f"{cell_seg.name}_nuc_fixed"]
    assert corrected_layer.data.shape == cell_seg.shape, (
        "Corrected layer shape is incorrect."
    )
    assert corrected_layer.data.dtype == cell_seg._data.dtype, (
        "Corrected layer data type is incorrect."
    )
