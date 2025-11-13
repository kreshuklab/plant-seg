import numpy as np
import pytest

from plantseg.functionals.proofreading import utils


def test_get_bboxes3D():
    segmentation = np.zeros((10, 10, 10), dtype=int)
    segmentation[:2, :2, :2] = 1
    segmentation[3:5, 3:6, 3:7] = 2
    labels_idx = [1, 2]

    bboxes = utils._get_bboxes3D(segmentation, labels_idx)

    assert all([k in bboxes for k in labels_idx])
    assert bboxes[1] == [[0, 0, 0], [1, 1, 1]]
    assert bboxes[2] == [[3, 3, 3], [4, 5, 6]]
