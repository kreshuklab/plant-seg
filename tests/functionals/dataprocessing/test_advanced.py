import numpy as np

from plantseg.functionals.dataprocessing.advanced_dataprocessing import remove_false_positives_by_foreground_probability


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
