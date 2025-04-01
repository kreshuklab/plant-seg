import argparse
import copy
import csv
import glob
import os
import time
import warnings
from datetime import datetime
from difflib import SequenceMatcher

import h5py
import numpy as np
import yaml
from rand import adapted_rand
from scipy.ndimage import zoom
from simple_hash import simple_hash
from voi import voi

# Add new metrics if needed
metrics = {"voi": voi, "adapted_rand": adapted_rand, "hash": simple_hash}


def load_config():
    parser = argparse.ArgumentParser(
        description="Instances Segmentation Evaluation Script"
    )
    parser.add_argument(
        "--config", type=str, help="Path to the YAML config file", required=True
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)
    return config


def create_result_placeholder(config, metrics):
    header = {"segmentation_file": None, "gt_file": None}

    # Metrics used
    for key in metrics.keys():
        header[key] = None

    # Custom keys for comments
    for key in config["metadata"].keys():
        header[key] = config["metadata"][key]

    if "train_config_file" in config:
        with open(config["train_config_file"], "r") as file:
            header["metadata_train"] = file.readlines()
    else:
        header["metadata_train"] = None

    if "segmentation_config_file" in config:
        with open(config["segmentation_config_file"], "r") as file:
            header["metadata_segmentation"] = file.readlines()
    else:
        header["metadata_segmentation"] = None

    return header


def automatic_file_matching(all_gt, all_seg):
    all_gt = np.array(all_gt)
    all_seg = np.array(all_seg)
    assignment_matrix = np.zeros((len(all_gt), len(all_seg)))
    for i, gt in enumerate(all_gt):
        for j, seg in enumerate(all_seg):
            assignment_matrix[i, j] = SequenceMatcher(None, gt, seg).ratio()

    assignment_matrix = np.argmax(assignment_matrix, axis=0)
    all_gt = all_gt[assignment_matrix]
    return zip(all_gt, all_seg)


def run_evaluation(gtarray, segarray, remove_background=True):
    timer = -time.time()
    # Check for problems in data types
    # double check for type and sign to allow a bit of slack in using _
    # int for segmentation and not only uint)
    if not np.issubdtype(segarray.dtype, np.integer):
        return None

    if not np.issubdtype(gtarray.dtype, np.integer):
        warnings.warn("Ground truth is not an integer array")
        return None

    if np.any(segarray < 0):
        warnings.warn("Found negative indices, segmentation must be positive")
        return None

    if np.any(gtarray < 0):
        warnings.warn("Found negative indices, ground truth must be positive")
        return None

    # Cast into uint3232
    segarray = segarray.astype(np.uint32)
    gtarray = gtarray.astype(np.uint32)

    # Resize segmentation to gt size for apple to apple comparison in the scores
    if segarray.shape != gtarray.shape:
        print(
            "- Segmentation shape:",
            segarray.shape,
            "Ground truth shape: ",
            gtarray.shape,
        )

        print("- Shape mismatch, trying to fixing it")
        factor = tuple(
            [
                g_shape / seg_shape
                for g_shape, seg_shape in zip(gtarray.shape, segarray.shape)
            ]
        )
        segarray = zoom(segarray, factor, order=0).astype(np.uint32)

    if remove_background:
        print("- Removing background")
        mask = gtarray != 0
        gtarray = gtarray[mask].ravel()
        segarray = segarray[mask].ravel()

    # Run all metric
    print("- Start evaluations")
    scores = {}
    for key, metric in metrics.items():
        result = metric(segarray.ravel(), gtarray.ravel())
        scores[key] = result
        print(key, ": ", result)

    timer += time.time()
    print("- Evaluation took %.1f s" % timer)
    return scores


def collect_results(header, scores, gt, seg):
    new_header = copy.deepcopy(header)
    new_header["gt_file"] = os.path.split(gt)[1]
    new_header["segmentation_file"] = os.path.split(seg)[1]

    for key, value in scores.items():
        new_header[key] = value
    return new_header


def parse_gt_seg_file_pairs(file_pairs, gt_dir, seg_dir):
    results = []
    for file_pair in file_pairs:
        # open H5 file containing segmentation results
        gt_file_path = os.path.join(gt_dir, file_pair["gt_filename"])
        seg_file_path = os.path.join(seg_dir, file_pair["seg_filename"])
        results.append((gt_file_path, seg_file_path))
    return results


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


if __name__ == "__main__":
    # read config
    eval_config = load_config()

    # Init cvs header and results container
    result_placeholder = create_result_placeholder(eval_config, metrics=metrics)
    results = []

    remove_background = (
        eval_config["remove_background"] if "remove_background" in eval_config else True
    )

    # Make sure that GT and segmentation directories are present in the FS
    # assert os.path.isdir(eval_config["gt_dir"]) and os.path.isdir(eval_config["seg_dir"])

    # Parse the files paths and return an iterable of tuples (gt_path, seg_path)
    if "files_pairs" in eval_config:
        # create (gt, seg) file pairs
        gt_seg_all = parse_gt_seg_file_pairs(
            eval_config["files_pairs"], eval_config["gt_dir"], eval_config["seg_dir"]
        )
    else:
        # List all files
        _all_gt = sorted(glob.glob(os.path.join(eval_config["gt_dir"], "*.h5")))
        _all_seg = sorted(glob.glob(eval_config["seg_dir"] + "/*.h5"))
        gt_seg_all = automatic_file_matching(_all_gt, _all_seg)

    # Run evaluation
    for _gt, _seg in gt_seg_all:
        print(f"Evaluating {_seg} with GT {_gt}...")
        # Load GT and segmentation into memory
        with h5py.File(_gt, "r") as gt:
            with h5py.File(_seg, "r") as seg:
                _segarray = seg[eval_config["seg_name"]][...]
                _gtarray = gt[eval_config["gt_name"]][...]

        _scores = run_evaluation(_gtarray, _segarray, remove_background)
        results.append(collect_results(result_placeholder, _scores, _gt, _seg))

    # Save CSV
    write_csv(eval_config["output_csv"], results)
