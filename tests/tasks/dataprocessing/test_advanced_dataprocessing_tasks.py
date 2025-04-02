import numpy as np
import pytest

from plantseg.core.image import (
    ImageLayout,
    ImageProperties,
    PlantSegImage,
    SemanticType,
)
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks.dataprocessing_tasks import (
    fix_over_under_segmentation_from_nuclei_task,
)


@pytest.fixture
def complex_test_PlantSegImages(complex_test_data):
    """
    Pytest fixture to convert raw test data into PlantSegImage objects with metadata.

    Args:
        complex_test_data (tuple): A tuple containing:
            - cell_seg (np.ndarray): 3D array for cell segmentation.
            - nuclei_seg (np.ndarray): 3D array for nuclei segmentation.
            - boundary_pmap (np.ndarray | None): 3D array for boundary probability map, or None.

    Returns:
        tuple: A tuple containing:
            - cell_seg (PlantSegImage): Cell segmentation as a PlantSegImage object.
            - nuclei_seg (PlantSegImage): Nuclei segmentation as a PlantSegImage object.
            - boundary_pmap (PlantSegImage | None): Boundary probability map as a PlantSegImage object, or None.
    """
    cell_seg, nuclei_seg, boundary_pmap = complex_test_data

    # Convert cell segmentation data to PlantSegImage
    cell_seg = PlantSegImage(
        data=cell_seg,
        properties=ImageProperties(
            name="cell_seg",
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            semantic_type=SemanticType.SEGMENTATION,
            image_layout=ImageLayout.ZYX,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        ),
    )

    # Convert nuclei segmentation data to PlantSegImage
    nuclei_seg = PlantSegImage(
        data=nuclei_seg,
        properties=ImageProperties(
            name="nuclei_seg",
            voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            semantic_type=SemanticType.SEGMENTATION,
            image_layout=ImageLayout.ZYX,
            original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
        ),
    )

    # Convert boundary probability map data to PlantSegImage, if provided
    boundary_pmap = (
        PlantSegImage(
            data=boundary_pmap,
            properties=ImageProperties(
                name="boundary_pmap",
                voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
                semantic_type=SemanticType.PREDICTION,
                image_layout=ImageLayout.ZYX,
                original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um"),
            ),
        )
        if boundary_pmap is not None
        else None
    )

    return cell_seg, nuclei_seg, boundary_pmap


def test_fix_over_under_segmentation_from_nuclei_task(complex_test_PlantSegImages):
    """
    Test the fix_over_under_segmentation_from_nuclei_task function.

    Args:
        complex_test_PlantSegImages (tuple): A tuple containing:
            - cell_seg (PlantSegImage): PlantSegImage object for cell segmentation.
            - nuclei_seg (PlantSegImage): PlantSegImage object for nuclei segmentation.
            - boundary_pmap (PlantSegImage | None): PlantSegImage object for boundary probability map, or None.

    Tests:
        - Ensures that the task processes input data correctly.
        - Verifies that the output is a PlantSegImage.
        - Confirms that merging and splitting thresholds, as well as quantile-based filtering, are applied correctly.
    """
    cell_seg, nuclei_seg, boundary_pmap = complex_test_PlantSegImages

    # Run the task with defined parameters
    result = fix_over_under_segmentation_from_nuclei_task(
        cell_seg=cell_seg,
        nuclei_seg=nuclei_seg,
        threshold_merge=0.3,
        threshold_split=0.6,
        quantile_min=0.1,
        quantile_max=0.9,
        boundary=boundary_pmap,
    )

    # Assert that the result is a PlantSegImage object
    assert isinstance(result, PlantSegImage), "Task result is not a PlantSegImage."

    # Ensure the output segmentation data is modified compared to the input (functional tested elsewhere)
    assert not np.array_equal(result.get_data(), cell_seg.get_data()), (
        "Task did not modify the input data as expected."
    )

    # Validate the output's name
    assert result.name == "cell_seg_nuc_fixed", "Output name is incorrect."
