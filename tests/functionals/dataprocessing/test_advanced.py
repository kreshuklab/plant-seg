import numpy as np
import pytest

from plantseg.functionals.dataprocessing.advanced_dataprocessing import (
    fix_over_under_segmentation_from_nuclei,
    remove_false_positives_by_foreground_probability,
)


@pytest.fixture
def complex_test_data():
    """
    Generates a complex 3D dataset with both under-segmented and over-segmented cells.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: cell segmentation, nuclei segmentation, and boundary probability map.
    """
    # Create a 3D grid of zeros
    cell_seg = np.zeros((10, 10, 10), dtype=np.uint16)
    nuclei_seg = np.zeros_like(cell_seg, dtype=np.uint16)

    # Define cells with under-segmentation (multiple nuclei in one cell)
    # Cell 1: covers (2, 2, 2) to (5, 5, 5), contains two nuclei
    cell_seg[2:6, 2:6, 2:6] = 1
    nuclei_seg[2:4, 2:3, 2:3] = 1
    nuclei_seg[4:6, 5:6, 5:6] = 2

    # Define cells with over-segmentation (one nucleus split into multiple cells)
    # Cell 2 and 3: cover (6, 6, 6) to (8, 8, 8), with one nucleus overlapping both cells
    cell_seg[6:8, 6:10, 6:10] = 2
    cell_seg[8:10, 6:10, 6:10] = 3
    nuclei_seg[7:9, 7:9, 7:9] = 3

    # Define another under-segmented region with a large cell and multiple nuclei
    # Cell 4: covers (1, 1, 6) to (3, 3, 8), contains two nuclei
    cell_seg[1:4, 1:4, 6:9] = 4
    nuclei_seg[1:2, 1:2, 6:7] = 4
    nuclei_seg[3:4, 3:4, 8:9] = 5

    # Generate a boundary probability map with higher values on the edges of the cells
    boundary_pmap = np.ones_like(cell_seg, dtype=np.float32)
    boundary_pmap[2:6, 2:6, 2:6] = 0.2
    boundary_pmap[6:8, 6:8, 6:8] = 0.2
    boundary_pmap[1:4, 1:4, 6:9] = 0.2

    return cell_seg, nuclei_seg, boundary_pmap


def test_remove_false_positives_by_foreground_probability():
    seg = np.ones((10, 10, 10), dtype=np.uint16)
    seg[2:8, 2:8, 2:8] += 20
    prob = np.zeros((10, 10, 10), dtype=np.float32)
    prob[2:8, 2:8, 2:8] += 0.4
    prob[3:7, 3:7, 3:7] += 0.4

    seg_new = remove_false_positives_by_foreground_probability(seg, prob, np.mean(prob[2:8, 2:8, 2:8] * 0.99))
    assert np.sum(seg_new == 1) == 216
    assert np.sum(seg_new == 2) == 0
    assert np.sum(seg_new == 0) == 1000 - 216

    seg_new = remove_false_positives_by_foreground_probability(seg, prob, np.mean(prob[2:8, 2:8, 2:8] * 1.01))
    assert np.sum(seg_new == 1) == 0
    assert np.sum(seg_new == 0) == 1000


def test_fix_over_under_segmentation_from_nuclei(complex_test_data):
    cell_seg, nuclei_seg, boundary_pmap = complex_test_data

    # Check that the input data is as expected
    assert len(np.unique(cell_seg[2:6, 2:6, 2:6])) == 1
    assert len(np.unique(cell_seg[1:4, 1:4, 6:9])) == 1
    assert len(np.unique(cell_seg[6:10, 6:10, 6:10])) == 2

    corrected_seg = fix_over_under_segmentation_from_nuclei(
        cell_seg=cell_seg,
        nuclei_seg=nuclei_seg,
        threshold_merge=0.3,
        threshold_split=0.6,
        quantiles_nuclei=(0.1, 0.9),
        boundary=boundary_pmap,
    )

    # Check under-segmented regions are split
    # Check that there are two unique labels in cell_seg[2:6, 2:6, 2:6]
    # Check that there are two unique labels in cell_seg[1:4, 1:4, 6:9]
    assert len(np.unique(corrected_seg[2:6, 2:6, 2:6])) == 2, "Undersegmentation not split."
    assert len(np.unique(corrected_seg[1:4, 1:4, 6:9])) == 2, "Undersegmentation not split."

    # Check over-segmented regions are merged
    # Check that there are 1 unique labels in cell_seg[6:8, 6:10, 6:10]
    assert len(np.unique(corrected_seg[6:10, 6:10, 6:10])) == 1, "Oversegmentation not merged."
