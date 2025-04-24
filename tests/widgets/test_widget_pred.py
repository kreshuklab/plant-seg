import numpy as np
import pytest
from magicgui import magicgui
from napari.types import LayerDataTuple

from plantseg.core.image import (
    ImageLayout,
    ImageProperties,
    PlantSegImage,
    SemanticType,
)
from plantseg.io.voxelsize import VoxelSize
from plantseg.viewer_napari.widgets.prediction import (
    UNetPredictionMode,
    widget_unet_prediction,
)


@pytest.fixture
def sample_image() -> PlantSegImage:
    return PlantSegImage(
        data=np.random.random((128, 128)),
        properties=ImageProperties(
            name="sample_image",
            semantic_type=SemanticType.RAW,
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
            image_layout=ImageLayout.YX,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0)),
        ),
    )


@magicgui
def widget_add_image(image: PlantSegImage) -> LayerDataTuple:
    """Add a plantseg.core.image.PlantSegImage to napari viewer as a napari.layers.Layer."""
    return image.to_napari_layer_tuple()


def test_widget_unet_prediction_advanced(qtbot, make_napari_viewer_proxy, sample_image):
    viewer = make_napari_viewer_proxy()
    widget_add_image(sample_image)

    model_name = "confocal_2D_unet_ovules_ds2x"
    count_layers = len(viewer.layers)
    widget_unet_prediction(
        image=viewer.layers[sample_image.name],
        mode=UNetPredictionMode.PLANTSEG,
        model_name=model_name,
        device="cpu",
        advanced=True,
        patch_size=(1, 96, 96),
        patch_halo=(0, 16, 16),
        single_patch=True,
        update_other_widgets=False,
    )
    qtbot.waitUntil(lambda: count_layers < len(viewer.layers), timeout=20000)

    print(viewer.layers)
    assert viewer.layers[-1].name == f"{sample_image.name}_{model_name}_0"


def test_widget_unet_prediction_advanced_default(qtbot):
    assert widget_unet_prediction.patch_size.value[0] != 1, (
        "Patch size should not be 1 by default"
    )

    model_name = "confocal_2D_unet_ovules_ds2x"
    widget_unet_prediction.model_name.value = model_name
    widget_unet_prediction.advanced.value = True
    qtbot.waitUntil(
        lambda: widget_unet_prediction.patch_size.value[0] != 128, timeout=30000
    )

    assert widget_unet_prediction.patch_size.value[0] == 1, (
        "Patch size should be 1 for 2D UNet"
    )
