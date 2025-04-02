import numpy as np
import pytest

from plantseg.core.image import (
    ImageLayout,
    ImageProperties,
    PlantSegImage,
    SemanticType,
)
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks.io_tasks import export_image_task, import_image_task


@pytest.mark.parametrize(
    "shape, layout, export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_image_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.rand(*shape).astype("float32")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        source_file_name="test",
    )
    image = PlantSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="float32",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image: PlantSegImage = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="raw",
        stack_layout=layout,
        m_slicing=None,
    )

    original_data = image.get_data()
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() <= 1.0  # check if the normalization is applied
    assert imported_data.max() <= 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


@pytest.mark.parametrize(
    "shape, layout, export_format",
    [
        ((32, 64, 64), ImageLayout.ZYX, "tiff"),
        ((32, 64, 64), ImageLayout.ZYX, "h5"),
        ((32, 64, 64), ImageLayout.ZYX, "zarr"),
        ((64, 64), ImageLayout.YX, "tiff"),
        ((64, 64), ImageLayout.YX, "h5"),
        ((64, 64), ImageLayout.YX, "zarr"),
    ],
)
def test_label_io_round_trip(tmp_path, shape, layout, export_format):
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PlantSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    if export_format == "tiff":
        file_path = tmp_path / "test.tiff"
        key = None
    elif export_format == "h5":
        file_path = tmp_path / "test.h5"
        key = "raw"
    else:
        file_path = tmp_path / "test.zarr"
        key = "raw"

    imported_image: PlantSegImage = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout=layout,
        m_slicing=None,
    )

    original_data = image.get_data()
    imported_data = imported_image.get_data()

    assert np.allclose(original_data, imported_data)
    assert original_data.max() > 1.0  # check if the normalization is not applied
    assert imported_data.max() > 1.0

    assert image.voxel_size == imported_image.voxel_size
    assert image.semantic_type == imported_image.semantic_type
    assert image.image_layout == imported_image.image_layout


def test_io_slicing_trip(tmp_path):
    shape = (32, 64, 64)
    layout = ImageLayout.ZYX
    export_format = "tiff"
    mock_data = np.random.randint(0, 10, size=shape).astype("uint16")

    property = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.SEGMENTATION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PlantSegImage(data=mock_data, properties=property)

    export_image_task(
        image=image,
        export_directory=tmp_path,
        name_pattern="test_raw",
        key="raw",
        export_format=export_format,
        data_type="uint16",
    )

    file_path = tmp_path / "test_raw.tiff"
    key = None

    imported_image: PlantSegImage = import_image_task(
        input_path=file_path,
        key=key,
        image_name="tesi_import",
        semantic_type="segmentation",
        stack_layout=layout,
        m_slicing="5:10,:, :50",
    )

    imported_data = imported_image.get_data()

    assert imported_data.shape == (5, 64, 50)
