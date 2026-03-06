import numpy as np
import pytest

from panseg.functionals.proofreading.split_merge_tools import _merge_from_seeds


def test_merge_from_seeds_all_to_one():
    """Test basic merging functionality"""
    # Create a simple 3D segmentation
    segmentation = np.array(
        [
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
            [[2, 1, 2], [1, 2, 1], [2, 1, 2]],
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {1: np.array([[0, 0, 0], [2, 2, 2]]), 2: np.array([[0, 0, 0], [2, 2, 2]])}

    all_idx = np.array([1, 2])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.ones_like(segmentation)
    np.testing.assert_array_equal(result_seg, expected_result)
    assert all([rs == slice(0, 3, None) for rs in result_slice])

    assert len(result_bboxes) == 1
    assert np.all(result_bboxes[1] == np.array([[0, 0, 0], [3, 3, 3]]))


def test_merge_from_seeds_with_other_label():
    """Test merging when 0 is in the indices"""
    # Create a simple 3D segmentation
    segmentation = np.array(
        [
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 0, 1], [0, 1, 0], [3, 3, 1]],
            [[0, 1, 0], [1, 0, 1], [3, 1, 0]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {
        0: np.array([[0, 0, 0], [3, 3, 3]]),
        1: np.array([[0, 0, 0], [3, 3, 3]]),
        3: np.array([[1, 2, 0], [3, 3, 2]]),
    }

    all_idx = np.array([0, 1])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.zeros_like(segmentation)
    expected_result[segmentation == 0] = 0
    expected_result[segmentation == 1] = 0
    expected_result[segmentation == 3] = 3

    np.testing.assert_array_equal(result_seg, expected_result)
    assert all([rs == slice(0, 3, None) for rs in result_slice])

    # Check that bboxes was updated with the new label
    assert np.array_equal(result_bboxes[0], region_bbox)
    assert len(result_bboxes) == 2
    assert np.all(result_bboxes[0] == np.array([[0, 0, 0], [3, 3, 3]]))
    assert np.all(result_bboxes[3] == bboxes[3])


def test_merge_from_seeds_single_label():
    """Test merging with a single label"""
    segmentation = np.array(
        [
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0, 0], [3, 3, 3]])

    bboxes = {3: np.array([[0, 0, 0], [3, 3, 3]])}

    all_idx = np.array([3])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape
    np.testing.assert_array_equal(result_seg, segmentation)

    assert 3 in result_bboxes
    assert len(result_bboxes) == 1
    assert np.array_equal(result_bboxes[3], region_bbox)


def test_merge_from_seeds_2d():
    """Test merging functionality with 2D segmentation"""
    segmentation = np.array(
        [
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1],
        ]
    )

    region_slice = (slice(0, 3), slice(0, 3))
    region_bbox = np.array([[0, 0], [2, 2]])

    bboxes = {1: np.array([[0, 0], [2, 2]]), 2: np.array([[0, 0], [2, 2]])}

    all_idx = np.array([1, 2])

    result_seg, result_slice, result_bboxes = _merge_from_seeds(
        segmentation, region_slice, region_bbox, bboxes, all_idx
    )

    assert result_seg.shape == segmentation.shape

    expected_result = np.zeros_like(segmentation)
    expected_result[segmentation == 1] = 1
    expected_result[segmentation == 2] = 1
    expected_result[(segmentation != 1) & (segmentation != 2)] = segmentation[
        (segmentation != 1) & (segmentation != 2)
    ]

    np.testing.assert_array_equal(result_seg, expected_result)

    assert 1 in result_bboxes
    assert np.array_equal(result_bboxes[1], region_bbox)
