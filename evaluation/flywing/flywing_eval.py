import argparse
import csv
import glob
import os
import shutil
import tempfile
from concurrent import futures
from os.path import expanduser
from subprocess import CalledProcessError, check_output

import h5py
import numpy as np
import wget
from PIL import Image


def relabel(tracks):
    labels = list(np.unique(tracks))
    if 0 in labels:
        labels.remove(0)

    if len(labels) >= 2**16:
        print(
            "Track graph contains %d distinct labels, can not be expressed in int16. Skipping evaluation."
            % len(labels)
        )
        raise RuntimeError()

    old_values = np.array(labels)
    new_values = np.arange(1, len(labels) + 1, dtype=np.uint16)

    values_map = np.arange(int(tracks.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    tracks = values_map[tracks]

    return tracks


def compute_seg_score(res_seg, gt_seg):
    SCRIPT_PATH = os.path.join(
        expanduser("~"), ".plantseg_models", "segtra_measure", "Linux", "SEGMeasure"
    )
    SCRIPT_URL = "https://github.com/maisli/tracking_evaluation/raw/master/segtra_measure/Linux/SEGMeasure"

    if not os.path.exists(SCRIPT_PATH):
        print("Downloading script from: ", SCRIPT_URL)
        out_dir = os.path.split(SCRIPT_PATH)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        wget.download(SCRIPT_URL, out=out_dir)
        # make the script executable
        os.chmod(SCRIPT_PATH, 0o0775)

    # holy cow, they need 16-bit encodings!
    if res_seg.max() >= 2**16:
        print("Converting res to int16... ")
        res_seg = relabel(res_seg)
    if gt_seg.max() >= 2**16:
        print("Converting gt to int16...")
        gt_seg = relabel(gt_seg)

    res_seg = res_seg.astype(np.uint16)
    gt_seg = gt_seg.astype(np.uint16)

    # create a temp dir
    dataset_dir = tempfile.mkdtemp()
    print("Using temp dir %s" % dataset_dir)

    try:
        res_dir = os.path.join(dataset_dir, "01_RES")
        gt_dir = os.path.join(dataset_dir, "01_GT", "SEG")

        os.makedirs(res_dir)
        os.makedirs(gt_dir)

        # store seg and gt as stack of tif files...
        assert res_seg.shape[0] == gt_seg.shape[0]

        # FORMAT:
        #
        # GT segmentation:
        #   * background 0
        #   * objects with IDs >=1, 16bit...
        #   -> this is what we already have
        #
        # RES segmentation:
        #   * background 0
        #   * objects with unique IDs >=1 in 2D, change between frames
        #     (hope this is not necessary, we will run out of IDs due to 16-bit
        #     encoding...)

        print("Preparing files for evaluation binaries...")
        for z in range(res_seg.shape[0]):
            res_outfile = os.path.join(res_dir, "mask%03d.tif" % z)
            gt_outfile = os.path.join(gt_dir, "man_seg%03d.tif" % z)

            res_im = Image.fromarray(res_seg[z].astype("uint16"))
            gt_im = Image.fromarray(gt_seg[z].astype("uint16"))
            res_im.save(res_outfile)
            gt_im.save(gt_outfile)

        print("Computing SEG score...")
        try:
            seg_output = check_output([SCRIPT_PATH, dataset_dir, "01"])
        except CalledProcessError as exc:
            print("Calling SEGMeasure failed: ", exc.returncode, exc.output)
            seg_score = 0
        else:
            seg_score = float(seg_output.split()[2])

        print("SEG score: %f" % seg_score)

    finally:
        shutil.rmtree(dataset_dir)

    return seg_score


def remove_small_labels(cells, num_pixel):
    unique, counts = np.unique(cells, return_counts=True)
    small_labels = unique[counts <= num_pixel]
    print("remove small labels: ", len(small_labels))

    cells = replace(cells, small_labels, np.zeros((len(small_labels)), dtype=np.uint64))
    return cells.astype(np.uint64)


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


######################################################################################################################

N_THREADS = 16
# should we get TRA score in addition to SEG score
eval_tra = False


def evaluate_seg(gt_file, seg_file, dataset_name, eval_tracking):
    with h5py.File(gt_file, "r") as f:
        mask = f["volumes/labels/ignore"][...]
        gt_tracks = f["volumes/labels/tracks"][...]
        gt_track_graph = f["graphs/track_graph"][...]  # noqa: F841

    with h5py.File(seg_file, "r") as f:
        seg_cells = f[dataset_name][...]
        seg_cells[mask == 1] = 0
        seg_cells = remove_small_labels(seg_cells, 3)
        seg_cells[mask == 1] = 0

        # seg_lineages = f['volumes/labels/lineages'][...]
        # seg_lineages[mask == 1] = 0

    seg_score = compute_seg_score(seg_cells, gt_tracks)
    if eval_tracking:
        tra_score = (
            0  # compute_tra_score(seg_cells, seg_lineages, gt_tracks, gt_track_graph)
        )
    else:
        tra_score = 0.0

    # write results to CSV
    report = {"SEG": seg_score, "TRA": tra_score}

    output_fields = report.keys()
    out_csv = os.path.splitext(seg_file)[0] + ".csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(output_fields)
        w.writerow([report[k] for k in output_fields])


def compute_mean_std(files):
    seg_scores = []
    tra_scores = []

    for in_file in glob.glob(files):
        print("Reading results from {}".format(in_file))
        with open(in_file, "r") as f:
            reader = csv.DictReader(f)
            result = list(reader)[0]
            seg_score = float(result["SEG"])
            seg_scores.append(seg_score)
            tra_score = float(result["TRA"])
            tra_scores.append(tra_score)
            print("SEG: {}, TRA: {}".format(seg_score, tra_score))

    seg_scores = np.array(seg_scores)
    tra_scores = np.array(tra_scores)

    return (
        np.mean(seg_scores),
        np.std(seg_scores),
        np.mean(tra_scores),
        np.std(tra_scores),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate FlyWing segmentation")
    parser.add_argument(
        "--gt-dir",
        type=str,
        help="Path to directory with the ground truth files",
        required=True,
    )
    parser.add_argument(
        "--seg-dir",
        type=str,
        help="Path to directory with the segmentation files",
        required=True,
    )
    parser.add_argument(
        "--seg-dataset",
        type=str,
        default="segmentation",
        help="Segmentation dataset inside the H5",
    )
    args = parser.parse_args()

    gt_files = list(glob.glob(os.path.join(args.gt_dir, "*.hdf")))
    seg_files = list(glob.glob(os.path.join(args.seg_dir, "*.h5")))
    gt_seg_map = {}
    for gt_file in gt_files:
        seg_file = None
        prefix = os.path.split(gt_file)[1]
        prefix = os.path.splitext(prefix)[0]
        for sf in seg_files:
            filename = os.path.split(sf)[1]
            if filename.startswith(prefix):
                seg_file = sf

        assert seg_file is not None

        gt_seg_map[gt_file] = seg_file

    print("Processing: ", gt_seg_map)

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        tasks = [
            tp.submit(evaluate_seg, gt_file, seg_file, args.seg_dataset, eval_tra)
            for gt_file, seg_file in gt_seg_map.items()
        ]

        results = [t.result() for t in tasks]

    # compute mean/stdev for per/pro movies
    for file_type in ["per", "pro"]:
        per_result_files = os.path.join(args.seg_dir, "{}*.csv".format(file_type))
        seg_mean, seg_std, tra_mean, tra_std = compute_mean_std(per_result_files)
        print(
            "{} movies: seg_mean {}, seg_std {}, tra_mean {}, tra_std {}".format(
                file_type, seg_mean, seg_std, tra_mean, tra_std
            )
        )
