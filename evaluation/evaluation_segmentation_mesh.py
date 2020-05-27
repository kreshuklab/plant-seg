import csv
import glob
import os
from datetime import datetime
import numpy as np

from rand import adapted_rand
from voi import voi

from plyfile import PlyData, PlyElement


def write_csv(output_path, results):
    assert len(results) > 0
    keys = results[0].keys()
    time_stamp = datetime.now().strftime("%d_%m_%y_%H%M%S")
    path_with_ts = os.path.splitext(output_path)[0] + '_' + time_stamp + '.csv'
    print(f'Saving results to {path_with_ts}...')
    with open(path_with_ts, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def fix_labels(ply_gt, ply_seg):
    """
    Remove extra vertices from the ground truth
    """
    size = len(ply_gt.elements[0]["x"])

    gt_x = np.array(ply_gt.elements[0]["x"])
    seg_x = np.array(ply_seg.elements[0]["x"])

    new_gt_label = np.zeros_like(seg_x)

    gt_label = np.array(ply_gt.elements[0]["label"])

    for i in range(size):
        if seg_x.shape[0] > i:
            if abs(gt_x[i] - seg_x[i]) < 1e-16:
                new_gt_label[i] = gt_label[i]

    new_gt_label = clean_gt(new_gt_label).astype(np.int)
    return new_gt_label


def clean_gt(gt):
    """
    Ignore boundary vertices and background from the evaluation
    """
    u_labels, counts = np.unique(gt, return_counts=True)
    bg_label = u_labels[np.argmax(counts)]

    gt = np.where(gt == -1, 0, gt)
    gt = np.where(gt == bg_label, 0, gt)
    return gt


if __name__ == "__main__":
    results = []
    data_path = "/home/lcerrone/leaf_evaluation/"
    datasets = ["C07", "C03", "Ox"]
    time_points = ["*T0*", "*T1*", "*T2*"]
    gt_id = "GT.ply"
    seg_ids = ["pred*.ply", "proj*.ply", "autoSegm.ply"]

    for dataset in datasets:
        for tp in time_points:

            gt_path = glob.glob(os.path.join(data_path, dataset, tp + gt_id))
            if len(gt_path) > 0:
                gt_path = gt_path[0]
                print("gt found: ", gt_path)

                with open(gt_path, 'rb') as f:
                    plydata_gt = PlyData.read(f)

                gt = np.array(plydata_gt.elements[0]["label"])

                gt = clean_gt(gt)

                for seg_id in seg_ids:
                    seg_path = glob.glob(os.path.join(data_path, dataset, tp + seg_id))[0]
                    print("  seg: ", seg_path)

                    with open(seg_path, 'rb') as f:
                        plydata_seg = PlyData.read(f)

                    seg = np.array(plydata_seg.elements[0]["label"])
                    seg = np.where(seg == -1, 0, seg)

                    if len(seg) != len(gt):
                        gt = fix_labels(plydata_gt, plydata_seg)

                    _rand = adapted_rand(seg, gt)
                    _voi = voi(seg, gt)
                    print("  scores: ", _rand, _voi[0], _voi[1])

                    result = {"dataset": dataset,
                              "time_point": tp,
                              "segmentation_id": seg_id,
                              "segmentation_path": seg_path,
                              "groundtruth_path": gt_path,
                              "adapted_rand": _rand,
                              "voi": _voi}

                    results.append(result)

                write_csv("segmentation_mesh_tmp", results)

    write_csv("segmentation_mesh", results)
