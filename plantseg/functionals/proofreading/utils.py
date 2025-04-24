import numba
import numpy as np


@numba.njit
def _get_bboxes3D(segmentation, labels_idx):
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        _x = np.array([[shape[0], shape[1], shape[2]], [0, 0, 0]])
        bboxes[idx] = _x

    for z in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                idx = segmentation[z, x, y]
                if idx > 0:
                    zmin, xmin, ymin = bboxes[idx][0]
                    zmax, xmax, ymax = bboxes[idx][1]

                    if z < zmin:
                        bboxes[idx][0, 0] = z
                    if x < xmin:
                        bboxes[idx][0, 1] = x
                    if y < ymin:
                        bboxes[idx][0, 2] = y

                    if z > zmax:
                        bboxes[idx][1, 0] = z
                    if x > xmax:
                        bboxes[idx][1, 1] = x
                    if y > ymax:
                        bboxes[idx][1, 2] = y
    return bboxes


@numba.njit
def _get_bboxes2D(segmentation, labels_idx):
    shape = segmentation.shape

    bboxes = {}
    for idx in labels_idx:
        _x = np.array([[shape[0], shape[1]], [0, 0]])
        bboxes[idx] = _x

    for x in range(shape[0]):
        for y in range(shape[1]):
            idx = segmentation[x, y]
            if idx > 0:
                xmin, ymin = bboxes[idx][0]
                xmax, ymax = bboxes[idx][1]

                if x < xmin:
                    bboxes[idx][0, 0] = x
                if y < ymin:
                    bboxes[idx][0, 1] = y

                if x > xmax:
                    bboxes[idx][1, 0] = x
                if y > ymax:
                    bboxes[idx][1, 1] = y
    return bboxes


def _get_bboxes(segmentation, labels_idx):
    if len(segmentation.shape) == 3:
        return _get_bboxes3D(segmentation, labels_idx)
    elif len(segmentation.shape) == 2:
        return _get_bboxes2D(segmentation, labels_idx)
    else:
        raise ValueError("Segmentation shape not supported")


def get_bboxes(segmentation, slack=(1, 3, 3)):
    segmentation = segmentation.astype("int64")
    labels_idx = np.unique(segmentation)
    bboxes = _get_bboxes(segmentation, labels_idx)

    slack = np.array(slack)

    if len(segmentation.shape) == 2:
        slack = slack[1:]

    bboxes_out = {}
    for key, values in bboxes.items():
        values[0] = np.maximum(values[0] - slack, np.zeros_like(slack))
        values[1] = np.minimum(values[1] + slack, segmentation.shape)
        bboxes_out[int(key)] = values
    return bboxes_out


def get_idx_slice(indices, bboxes_dict: dict):
    if isinstance(indices, int):
        indices = [indices]

    list_min_values = [bboxes_dict[idx][0] for idx in indices]
    list_max_values = [bboxes_dict[idx][1] for idx in indices]
    min_values = np.min(np.stack(list_min_values), axis=0)
    max_values = np.max(np.stack(list_max_values), axis=0)
    values = np.stack([min_values, max_values])

    return tuple([slice(_min, _max) for _min, _max in zip(*values)]), values, min_values
