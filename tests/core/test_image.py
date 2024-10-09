from uuid import uuid4

import numpy as np
from napari.layers import Image

from plantseg.core.image import (
    ImageDimensionality,
    ImageLayout,
    ImageProperties,
    ImageType,
    PlantSegImage,
    SemanticType,
)
from plantseg.io.voxelsize import VoxelSize


# Tests for Enum classes
def test_semantic_type_enum():
    assert SemanticType.RAW.value == "raw"
    assert SemanticType.SEGMENTATION.value == "segmentation"
    assert SemanticType.PREDICTION.value == "prediction"
    assert SemanticType.LABEL.value == "label"


def test_image_type_enum():
    assert ImageType.IMAGE.value == "image"
    assert ImageType.LABEL.value == "labels"
    assert ImageType.to_choices() == ["image", "labels"]


def test_image_dimensionality_enum():
    assert ImageDimensionality.TWO.value == "2D"
    assert ImageDimensionality.THREE.value == "3D"


def test_image_layout_enum():
    assert ImageLayout.YX.value == "YX"
    assert ImageLayout.CYX.value == "CYX"
    assert ImageLayout.ZYX.value == "ZYX"
    assert ImageLayout.CZYX.value == "CZYX"
    assert ImageLayout.ZCYX.value == "ZCYX"
    assert ImageLayout.to_choices() == ["YX", "CYX", "ZYX", "CZYX", "ZCYX"]


# Tests for ImageProperties class
def test_image_properties_initialization():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert image_props.name == "test_image"
    assert image_props.semantic_type == SemanticType.RAW
    assert image_props.voxel_size == voxel_size
    assert image_props.image_layout == ImageLayout.ZYX
    assert image_props.original_voxel_size == voxel_size


def test_image_properties_dimensionality():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    image_props_2d = ImageProperties(
        name="2D_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.YX,
        original_voxel_size=voxel_size,
    )
    assert image_props_2d.dimensionality == ImageDimensionality.TWO

    image_props_3d = ImageProperties(
        name="3D_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert image_props_3d.dimensionality == ImageDimensionality.THREE


def test_image_properties_image_type():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    raw_image_props = ImageProperties(
        name="raw_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert raw_image_props.image_type == ImageType.IMAGE

    label_image_props = ImageProperties(
        name="label_image",
        semantic_type=SemanticType.LABEL,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert label_image_props.image_type == ImageType.LABEL


def test_image_properties_channel_axis():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    cyx_image_props = ImageProperties(
        name="cyx_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.CYX,
        original_voxel_size=voxel_size,
    )
    assert cyx_image_props.channel_axis == 0

    zcyx_image_props = ImageProperties(
        name="zcyx_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZCYX,
        original_voxel_size=voxel_size,
    )
    assert zcyx_image_props.channel_axis == 1


def test_image_properties_interpolation_order():
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    label_image_props = ImageProperties(
        name="label_image",
        semantic_type=SemanticType.LABEL,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert label_image_props.interpolation_order() == 0

    raw_image_props = ImageProperties(
        name="raw_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    assert raw_image_props.interpolation_order() == 1


# Tests for PlantSegImage class
def test_plantseg_image_initialization():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)

    assert ps_image.shape == (10, 10, 10)
    assert ps_image.voxel_size == voxel_size
    assert ps_image.name == "test_image"


def test_plantseg_image_derive_new():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)

    new_data = np.random.rand(10, 10, 10)
    new_image = ps_image.derive_new(new_data, name="new_image")

    assert new_image.name == "new_image"
    assert new_image.shape == (10, 10, 10)
    assert new_image.voxel_size == voxel_size
    assert new_image.original_voxel_size == voxel_size


def test_plantseg_image_get_data():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)

    # Test without normalization
    normalised_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12)
    retrieved_data = ps_image.get_data(normalize_01=True)
    assert normalised_data.dtype == retrieved_data.dtype
    np.testing.assert_allclose(retrieved_data, normalised_data)

    # Test with normalization
    retrieved_data = ps_image.get_data(normalize_01=False)
    assert normalised_data.dtype == retrieved_data.dtype
    np.testing.assert_allclose(retrieved_data, data)


def test_plantseg_image_from_napari_layer():
    data = np.random.rand(10, 10, 10)
    voxel_size = (1.0, 1.0, 1.0)
    metadata = {
        "semantic_type": "raw",
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "ZYX",
        "id": uuid4(),
    }
    napari_layer = Image(data, metadata=metadata, name="test_image")

    ps_image = PlantSegImage.from_napari_layer(napari_layer)
    assert ps_image.name == "test_image"
    assert ps_image.shape == (10, 10, 10)
    assert ps_image.voxel_size.voxels_size == voxel_size
    assert tuple(ps_image.voxel_size) == voxel_size


def test_plantseg_image_to_napari_layer_tuple():
    data = np.random.rand(2, 2, 2)
    voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="test_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)
    layer_tuple = ps_image.to_napari_layer_tuple()

    assert isinstance(layer_tuple, tuple)
    layer_tuple = tuple(layer_tuple)
    np.testing.assert_allclose(layer_tuple[0], ps_image.get_data(normalize_01=True))
    assert "metadata" in layer_tuple[1]
    assert layer_tuple[2] == ps_image.image_type.value


def test_plantseg_image_scale_property():
    data = np.random.rand(10, 10, 10)
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)
    assert ps_image.scale == (0.5, 1.0, 1.0)


def test_requires_scaling():
    data = np.random.rand(10, 10, 10)

    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    same_voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    original_voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    assert same_voxel_size == voxel_size
    assert original_voxel_size != voxel_size

    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=original_voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)
    assert ps_image.requires_scaling is True

    image_props = ImageProperties(
        name="scaled_image",
        semantic_type=SemanticType.RAW,
        voxel_size=voxel_size,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=same_voxel_size,
    )
    ps_image = PlantSegImage(data, image_props)
    assert ps_image.requires_scaling is False

    assert same_voxel_size == voxel_size
    assert original_voxel_size != voxel_size
