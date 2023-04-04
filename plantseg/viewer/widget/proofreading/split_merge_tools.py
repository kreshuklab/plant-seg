import numpy as np
from skimage.segmentation import watershed

from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.proofreading.utils import get_bboxes, get_idx_slice


def _merge_from_seeds(segmentation, region_slice, region_bbox, bboxes, all_idx):
    region_segmentation = segmentation[region_slice]

    mask = [region_segmentation == idx for idx in all_idx]

    new_label = 0 if 0 in all_idx else all_idx[0]

    mask = np.logical_or.reduce(mask)
    region_segmentation[mask] = new_label
    bboxes[new_label] = region_bbox
    napari_formatted_logging('Merge complete', thread='Proofreading tool')
    return region_segmentation, region_slice, bboxes


def _split_from_seed(segmentation, sz, sx, sy, region_slice, all_idx, offsets, bboxes, image, seeds_values, max_label):
    local_sz, local_sx, local_sy = sz - offsets[0], sx - offsets[1], sy - offsets[2]

    region_image = image[region_slice]
    region_segmentation = segmentation[region_slice]

    region_seeds = np.zeros_like(region_segmentation)

    seeds_values += max_label
    region_seeds[local_sz, local_sx, local_sy] = seeds_values

    mask = [region_segmentation == idx for idx in all_idx]
    mask = np.logical_or.reduce(mask)

    new_seg = watershed(region_image, region_seeds, mask=mask, compactness=0.001)
    new_seg[~mask] = region_segmentation[~mask]

    new_bboxes = get_bboxes(new_seg)
    for idx in np.unique(seeds_values):
        values = new_bboxes[idx]
        values = values + offsets[None, :]
        bboxes[idx] = values

    napari_formatted_logging('Split complete', thread='Proofreading tool')
    return new_seg, region_slice, bboxes


def split_merge_from_seeds(seeds, segmentation, image, bboxes, max_label, correct_labels):
    # find seeds location ad label value
    sz, sx, sy = np.nonzero(seeds)

    seeds_values = seeds[sz, sx, sy]
    seeds_idx = np.unique(seeds_values)

    all_idx = segmentation[sz, sx, sy]
    all_idx = np.unique(all_idx)

    region_slice, region_bbox, offsets = get_idx_slice(all_idx, bboxes_dict=bboxes)

    correct_cell_idx = [idx for idx in all_idx if idx in correct_labels]
    if correct_cell_idx:
        napari_formatted_logging(f'Label {correct_cell_idx} is in the correct labels list. Cannot be modified',
                                 thread='Proofreading tool')
        return segmentation[region_slice], region_slice, bboxes

    if len(seeds_idx) == 1:
        return _merge_from_seeds(segmentation,
                                 region_slice,
                                 region_bbox,
                                 bboxes,
                                 all_idx)
    else:
        return _split_from_seed(segmentation,
                                sz, sx, sy,
                                region_slice,
                                all_idx,
                                offsets,
                                bboxes,
                                image,
                                seeds_values,
                                max_label)
