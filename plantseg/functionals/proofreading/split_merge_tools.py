import logging

import numpy as np
from skimage.segmentation import watershed

from plantseg.functionals.proofreading.utils import get_bboxes, get_idx_slice

logger = logging.getLogger(__name__)


def _merge_from_seeds(segmentation, region_slice, region_bbox, bboxes, all_idx):
    region_segmentation = segmentation[region_slice]

    mask = [region_segmentation == idx for idx in all_idx]

    new_label = 0 if 0 in all_idx else all_idx[0]

    mask = np.logical_or.reduce(mask)
    region_segmentation[mask] = new_label
    bboxes[new_label] = region_bbox
    logger.info("Merge complete")
    return region_segmentation, region_slice, bboxes


def _split_from_seed(
    segmentation,
    seeds_list,
    region_slice,
    all_idx,
    offsets,
    bboxes,
    image,
    seeds_values,
    max_label,
):
    local_seeds_list = [ls - of for ls, of in zip(seeds_list, offsets)]

    region_image = image[region_slice]
    region_segmentation = segmentation[region_slice]

    region_seeds = np.zeros_like(region_segmentation)

    seeds_values += max_label
    region_seeds[*local_seeds_list] = seeds_values

    mask = [region_segmentation == idx for idx in all_idx]
    mask = np.logical_or.reduce(mask)

    new_seg = watershed(region_image, region_seeds, mask=mask, compactness=0.001)
    new_seg[~mask] = region_segmentation[~mask]

    new_bboxes = get_bboxes(new_seg)
    for idx in np.unique(seeds_values):
        values = new_bboxes[idx]
        values = values + offsets[None, :]
        bboxes[idx] = values

    logger.info("Split complete")
    return new_seg, region_slice, bboxes


def split_merge_from_seeds(
    seeds, segmentation, image, bboxes, max_label, correct_labels
):
    # find seeds location ad label value
    seeds_list = np.nonzero(seeds)

    seeds_values = seeds[*seeds_list]
    seeds_idx = np.unique(seeds_values)

    all_idx = segmentation[*seeds_list]
    all_idx = np.unique(all_idx)

    region_slice, region_bbox, offsets = get_idx_slice(all_idx, bboxes_dict=bboxes)

    correct_cell_idx = [idx for idx in all_idx if idx in correct_labels]
    if correct_cell_idx:
        logger.info(
            f"Label {correct_cell_idx} is in the correct labels list. Cannot be modified"
        )
        return segmentation[region_slice], region_slice, bboxes

    if len(seeds_idx) == 1:
        return _merge_from_seeds(
            segmentation, region_slice, region_bbox, bboxes, all_idx
        )
    else:
        return _split_from_seed(
            segmentation,
            seeds_list,
            region_slice,
            all_idx,
            offsets,
            bboxes,
            image,
            seeds_values,
            max_label,
        )
