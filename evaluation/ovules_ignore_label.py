import argparse
import glob
import os

import h5py
import numpy as np
from scipy.ndimage import binary_dilation


def _ignore_unlabeled(label, seg, ignore_label=-1):
    """
    Given the ground truth 'label' image and the segmentation from PlantSeg 'seg'
    the function assigns 'ignore_label' to the unlabeled regions in the ovule ground truth.

    Returns:
         ground truth array where the voxels from the unlabeled regions are assigned 'ignore_label'
    """
    # copy the original ground truth labels
    result = label.astype(np.int16, copy=True)
    # get the mask for ground truth instances
    gt_mask = label > 0
    # dilate te ground truth mask
    gt_mask = binary_dilation(gt_mask)

    # zero out ground truth mask in seg volume
    seg[gt_mask] = 0
    # get the ignore mask (we choose everything bigger than 1, cause the output from PlantSeg assigns 1 to the background)
    ignore_mask = seg > 1
    # dilate the ignore mask
    ignore_mask = binary_dilation(ignore_mask)
    # assign ignore label to ignore_mask in the resulting array
    result[ignore_mask] = ignore_label
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add ignore label to ovules dataset")
    parser.add_argument(
        "--gt_dir", type=str, help="Path to the ground truth directory", required=True
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        help="Path to the PlantSeg segmentation directory",
        required=True,
    )

    args = parser.parse_args()
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.h5")))
    seg_files = sorted(glob.glob(os.path.join(args.seg_dir, "*.h5")))

    for gt_file, seg_file in zip(gt_files, seg_files):
        print(f"GT file: {gt_file}, seg file: {seg_file}")
        with h5py.File(gt_file, "r") as f:
            label = f["label"][...]

        with h5py.File(seg_file, "r") as f:
            seg = f["segmentation"][...]

        gt_with_ignore = _ignore_unlabeled(label, seg)

        with h5py.File(gt_file, "r+") as f:
            f.create_dataset(
                "label_with_ignore", data=gt_with_ignore, compression="gzip"
            )
