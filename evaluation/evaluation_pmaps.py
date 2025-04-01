import argparse
import csv
from datetime import datetime

import h5py
import numpy as np
from scipy.ndimage import zoom
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from sklearn.metrics import precision_score, recall_score


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


def write_csv(output_path, results):
    assert len(results) > 0
    keys = results[0].keys()
    time_stamp = datetime.now().strftime("%d_%m_%y_%H%M%S")
    path_with_ts = os.path.splitext(output_path)[0] + "_" + time_stamp + ".csv"
    print(f"Saving results to {path_with_ts}...")
    with open(path_with_ts, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def parse():
    parser = argparse.ArgumentParser(description="Pmaps Quality Evaluation Script")
    parser.add_argument(
        "--gt",
        type=str,
        help="Path to directory with the ground truth files",
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to directory with the predictions files",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        help="thresholds at which the predictions will be binarized",
        required=True,
    )
    parser.add_argument(
        "--out-file",
        type=str,
        help="define name (and location) of output file (final name: out-file + timestamp + .csv)",
        required=False,
        default="pmaps_evaluation",
    )
    parser.add_argument(
        "--p-key",
        type=str,
        default="predictions",
        help="predictions dataset name inside h5",
        required=False,
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        default="label",
        help="ground truth dataset name inside h5",
        required=False,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="must match the default smoothing used in training. Default ovules 1.3",
        required=False,
    )
    args = parser.parse_args()
    return args


def pmaps_evaluation(
    gt_path,
    predictions_path,
    thresholds,
    out_name="pmaps_evaluation",
    p_key="predictions",
    gt_key="label",
    sigma=1.0,
):
    if isinstance(thresholds, float):
        assert 0 < thresholds < 1, "threshold must be a float between 0 and 1."
        thresholds = [thresholds]

    elif isinstance(thresholds, list):
        for _t in thresholds:
            assert 0 < _t < 1, "each threshold must be a float between 0 and 1."
    else:
        TypeError("thresholds type not understood")

    all_predictions, all_gt = [], []
    if os.path.isdir(gt_path) and os.path.isdir(predictions_path):
        print(
            "Correct ordering is not guaranteed!!! Please check the correctness at each run."
        )
        all_gt = sorted(glob.glob(gt_path + "/*.h5"))
        all_predictions = sorted(glob.glob(predictions_path + "/*.h5"))
        assert len(all_gt) == len(all_predictions), (
            "ground truth and predictions must have same length."
        )
    elif os.path.isfile(gt_path) and os.path.isfile(predictions_path):
        all_gt = [gt_path]
        all_predictions = [predictions_path]

    else:
        NotImplementedError(
            "gt and predictions inputs must be directories or single files. Moreover, types must match."
        )

    results = []
    for pmap_file, gt_file in zip(all_predictions, all_gt):
        print("Processing (gt, pmap): ", gt_file, pmap_file)
        with h5py.File(gt_file, "r") as gt_f:
            with h5py.File(pmap_file, "r") as pmap_f:
                print("seg shape, gt shape: ", pmap_f[p_key].shape, gt_f[gt_key].shape)
                pmap = pmap_f[p_key][0, ...]
                gt = gt_f[gt_key][...]

        # Resize segmentation to gt size for apple to apple comparison in the scores
        if gt.shape != pmap.shape:
            factor = tuple(
                [
                    g_shape / seg_shape
                    for g_shape, seg_shape in zip(gt.shape, pmap.shape)
                ]
            )
            pmap = zoom(pmap, factor)

        # generate gt boundaries
        boundaries = find_boundaries(gt, connectivity=2, mode="thick").astype("uint8")

        for threshold in thresholds:
            _pmap = np.zeros_like(pmap)
            # binarize predictions
            _pmap[pmap >= threshold] = 1
            _pmap[pmap < threshold] = 0

            # Measure accuracy
            mask = _pmap == boundaries
            accuracy = np.sum(mask) / mask.size

            # Measure scores
            precision = precision_score(boundaries.ravel(), _pmap.ravel())
            recall = recall_score(boundaries.ravel(), _pmap.ravel())
            f1 = 2 * ((precision * recall) / (precision + recall))

            print(
                f"threshold: {threshold:0.2f},"
                f" accuracy: {accuracy:0.3f},"
                f" f1 score: {f1:0.3f},"
                f" precision: {precision:0.3f},"
                f" recall: {recall:0.3f}"
            )

            results.append(
                {
                    "threshold": threshold,
                    "gt": gt_file,
                    "pmap": pmap_file,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1 score": f1,
                }
            )
    write_csv(out_name, results)


if __name__ == "__main__":
    import glob
    import os

    args = parse()
    pmaps_evaluation(
        args.gt,
        args.predictions,
        args.threshold,
        out_name=args.out_file,
        p_key=args.p_key,
        gt_key=args.gt_key,
        sigma=args.sigma,
    )
