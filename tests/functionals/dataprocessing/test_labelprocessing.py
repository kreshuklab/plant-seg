import numpy as np

from plantseg.functionals.dataprocessing.labelprocessing import relabel_segmentation, set_background_to_value


class TestDataProcessing:
    def test_set_background_to_value(self):
        segmentation = np.ones((10, 10))
        segmentation[1, 1] = 2
        segmentation[5, 5] = 3
        new_segmentation = set_background_to_value(segmentation, 0)
        assert np.allclose(np.unique(new_segmentation), [0, 3, 4])


# Test relabel_segmentation
def test_relabel_segmentation():
    # Case 1: Simple 2D segmentation with contiguous regions
    segmentation_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 2, 2],
            [0, 0, 2, 2],
        ]
    )

    relabeled_image = relabel_segmentation(segmentation_image)

    # Expected relabeling
    expected_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 2, 2],
            [0, 0, 2, 2],
        ]
    )

    assert np.array_equal(relabeled_image, expected_image)

    # Case 2: 2D segmentation with non-contiguous regions with 1-connectivity
    segmentation_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )

    relabeled_image = relabel_segmentation(segmentation_image)
    print(relabeled_image)

    # Expected relabeling: the two 2-connectivity-connected regions should not be relabeled differently
    assert relabeled_image[0, 0] == 1
    assert relabeled_image[3, 3] == 1  # Becuase PlantSeg uses 2-connectivity by default
    np.testing.assert_allclose(np.unique(relabeled_image), [0, 1])

    # Case 3: 2D segmentation with non-contiguous regions with 2-connectivity
    segmentation_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]
    )

    relabeled_image = relabel_segmentation(segmentation_image)

    # Expected relabeling: the two separate regions labeled "1" should be relabeled differently
    assert relabeled_image[0, 0] == 1
    assert relabeled_image[3, 3] == 2
    np.testing.assert_allclose(np.unique(relabeled_image), [0, 1, 2])


# Test set_background_to_value
def test_set_background_to_value():
    # Case 1: Simple 2D segmentation with one clear background
    segmentation_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 2, 2],
            [0, 0, 2, 2],
        ]
    )

    new_segmentation_image = set_background_to_value(segmentation_image, value=3)

    # Expected output: background (label 0) should be set to 3
    expected_image = np.array(
        [
            [1, 1, 3, 3],
            [1, 1, 2, 2],
            [3, 3, 2, 2],
        ]
    )

    assert np.array_equal(new_segmentation_image, expected_image)

    # Case 2: Background is not the largest label
    segmentation_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 2, 2],
            [2, 2, 2, 2],
        ]
    )

    new_segmentation_image = set_background_to_value(segmentation_image, value=5)

    # Expected output: label 2 should be set to 5 because it's the largest segment
    expected_image = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 5, 5],
            [5, 5, 5, 5],
        ]
    )

    assert np.array_equal(new_segmentation_image, expected_image)

    # Case 3: 3D segmentation example
    segmentation_image = np.array(
        [
            [
                [1, 1, 0],
                [0, 0, 2],
            ],
            [
                [1, 1, 2],
                [2, 2, 2],
            ],
        ]
    )

    new_segmentation_image = set_background_to_value(segmentation_image, value=7)

    # Expected output: label 2 should be set to 7 because it's the largest segment
    expected_image = np.array(
        [
            [
                [1, 1, 0],
                [0, 0, 7],
            ],
            [
                [1, 1, 7],
                [7, 7, 7],
            ],
        ]
    )

    assert np.array_equal(new_segmentation_image, expected_image)
