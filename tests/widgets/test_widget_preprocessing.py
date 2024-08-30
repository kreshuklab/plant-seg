import napari
import numpy as np
import pytest
from magicgui import magicgui
from napari.types import LayerDataTuple

from plantseg.core.image import ImageLayout, ImageProperties, PlantSegImage, SemanticType
from plantseg.core.voxelsize import VoxelSize
from plantseg.viewer_napari.widgets.dataprocessing import RescaleModes, widget_rescaling


def create_layer_name(name: str, suffix: str):
    return f"{name}_{suffix}"


@pytest.fixture
def sample_image() -> PlantSegImage:
    return PlantSegImage(
        data=np.random.random((10, 100, 100)),
        properties=ImageProperties(
            name="sample_image",
            semantic_type=SemanticType.RAW,
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
            image_layout=ImageLayout.ZYX,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
        ),
    )


@pytest.fixture
def sample_label() -> PlantSegImage:
    return PlantSegImage(
        data=np.random.randint(0, 2, (5, 100, 100)),
        properties=ImageProperties(
            name="sample_label",
            semantic_type=SemanticType.RAW,
            voxel_size=VoxelSize(voxels_size=(2.0, 1.0, 1.0)),
            image_layout=ImageLayout.ZYX,
            original_voxel_size=VoxelSize(voxels_size=(2.0, 1.0, 1.0)),
        ),
    )


@magicgui
def widget_add_image(image: PlantSegImage) -> LayerDataTuple:
    """Add a plantseg.core.image.PlantSegImage to napari viewer as a napari.layers.Layer."""
    return image.to_napari_layer_tuple()


class TestWidgetRescaling:
    def test_rescaling_from_factor(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        widget_add_image(sample_image)

        factor = 0.5
        widget_rescaling(
            image=viewer.layers[sample_image.name],
            mode=RescaleModes.FROM_FACTOR,
            rescaling_factor=(factor, factor, factor),
            update_other_widgets=False,
        )
        napari.run()

        old_layer = viewer.layers[sample_image.name]
        new_layer = viewer.layers[sample_image.name + '_rescaled']
        np.testing.assert_allclose(new_layer.data.shape, np.multiply(old_layer.data.shape, factor), rtol=1e-5)
        np.testing.assert_allclose(np.multiply(new_layer.scale, factor), old_layer.scale, rtol=1e-5)
