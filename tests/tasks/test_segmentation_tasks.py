import numpy as np
import pytest

from plantseg.core.image import (
    ImageLayout,
    ImageProperties,
    PlantSegImage,
    SemanticType,
)
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks.segmentation_tasks import (
    clustering_segmentation_task,
    dt_watershed_task,
)


@pytest.mark.parametrize(
    "shape, layout, stacked, is_nuclei, clustering, mode",
    [
        ((32, 64, 64), ImageLayout.ZYX, False, False, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, True, False, "-"),
        ((32, 64, 64), ImageLayout.ZYX, False, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "gasp"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "multicut"),
        ((32, 64, 64), ImageLayout.ZYX, True, False, True, "mutex_ws"),
        ((64, 64), ImageLayout.YX, False, False, False, "-"),
        ((64, 64), ImageLayout.YX, False, True, False, "-"),
        ((64, 64), ImageLayout.YX, False, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, True, False, True, "gasp"),
        ((64, 64), ImageLayout.YX, False, False, True, "multicut"),
        ((64, 64), ImageLayout.YX, False, False, True, "mutex_ws"),
    ],
)
def test_dt_watershed_and_clustering(
    shape, layout, stacked, is_nuclei, clustering, mode
):
    mock_data = np.random.rand(*shape).astype("float32")

    property_ = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PlantSegImage(data=mock_data, properties=property_)

    result = dt_watershed_task(image=image, stacked=stacked, is_nuclei_image=is_nuclei)

    assert result.semantic_type == SemanticType.SEGMENTATION
    assert result.image_layout == property_.image_layout
    assert result.voxel_size == property_.voxel_size
    assert result.shape == mock_data.shape

    if not clustering:
        return None

    result_clustering = clustering_segmentation_task(
        image=image, over_segmentation=result, mode=mode
    )
    assert result_clustering.semantic_type == SemanticType.SEGMENTATION
    assert result_clustering.image_layout == property_.image_layout
    assert result_clustering.voxel_size == property_.voxel_size
    assert result_clustering.shape == mock_data.shape


def test_mutex():
    mock_data = np.random.rand(32, 64, 64).astype("float32")

    property_ = ImageProperties(
        name="test",
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        semantic_type=SemanticType.PREDICTION,
        image_layout=ImageLayout.ZYX,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
    )
    image = PlantSegImage(data=mock_data, properties=property_)

    result_clustering = clustering_segmentation_task(
        image=image, over_segmentation=None, mode="mutex_ws"
    )
    assert result_clustering.semantic_type == SemanticType.SEGMENTATION
    assert result_clustering.image_layout == property_.image_layout
    assert result_clustering.voxel_size == property_.voxel_size
    assert result_clustering.shape == mock_data.shape
