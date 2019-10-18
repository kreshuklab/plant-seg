import h5py
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from scipy.ndimage import zoom
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime
import csv


seg_key = "predictions"
gt_key = "label"
sigma = 1


def blur_boundary(boundary, sigma):
    boundary = gaussian(boundary, sigma=sigma)
    boundary[boundary >= 0.5] = 1
    boundary[boundary < 0.5] = 0
    return boundary


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


if __name__ == "__main__":
    import glob
    import os

    # Change those
    """
    name_files = ["ds1_unet",
                  "ds2_unet",
                  "ds3_unet",
                  "resunet_ds1x",
                  "resunet_ds2x",
                  "resunet_ds3x",
                  "unet_ds2x_dice",
                  "unet_ds2x_dice_lr2",
                  "unet_ds2x_dice_no_wd",
                  "unet_ds2x_lr2",
                  "unet_ds2x_no_wd",
                  "unet_ds2x_wd_00001"]

    all_paths = ["ds1_predictions/unet_ds1x",
                 "ds2_predictions/unet_ds2x",
                 "ds3_predictions/unet_ds3x",
                 "ds1_predictions/resunet_ds1x",
                 "ds2_predictions/resunet_ds2x",
                 "ds3_predictions/resunet_ds3x",
                 "ds2_predictions/unet_ds2x_dice",
                 "ds2_predictions/unet_ds2x_dice_lr2",
                 "ds2_predictions/unet_ds2x_dice_no_wd",
                 "ds2_predictions/unet_ds2x_lr2",
                 "ds2_predictions/unet_ds2x_no_wd",
                 "ds2_predictions/unet_ds2x_wd_00001"]
    """

    name_files = ["ds_dice_aff"]
    all_paths = ["ds2_predictions/unet_ds2x_bce_aff"]

    for file_name, path in zip(name_files, all_paths):
        all_gt = sorted(glob.glob("/mnt/localdata0/lcerrone/FOR_paper/TUM_Ovules_Dataset/Ovules_test_final/ds2_predictions/*.h5"))
        all_seg = sorted(glob.glob("/mnt/localdata0/lcerrone/FOR_paper/TUM_Ovules_Dataset/Ovules_test_final/" + path + "/*ons.h5"))

        results = []
        for pmap_file, gt_file in zip(all_seg, all_gt):
            print("Processing (gt, pmap): ", gt_file, pmap_file)
            with h5py.File(gt_file, 'r') as gt_f:
                with h5py.File(pmap_file, 'r') as pmap_f:
                    print("seg shape, gt shape: ", pmap_f[seg_key].shape, gt_f[gt_key].shape)
                    pmap = pmap_f[seg_key][0, ...]
                    gt = gt_f[gt_key][...]

            # Resize segmentation to gt size for apple to apple comparison in the scores
            if gt.shape != pmap.shape:
                factor = tuple([g_shape / seg_shape for g_shape, seg_shape in zip(gt.shape, pmap.shape)])
                pmap = zoom(pmap, factor)

            boundaries = find_boundaries(gt, connectivity=2)
            boundaries = blur_boundary(boundaries, sigma)

            pmap[pmap >= 0.5] = 1
            pmap[pmap < 0.5] = 0

            mask = (pmap == boundaries)
            accuracy = (mask.sum()/mask.size)
            print("accuracy: ", accuracy)

            f1 = f1_score(boundaries.ravel(), pmap.ravel())

            print("f1 score: ", f1)
            results.append({"gt": gt_file, "pmap": pmap_file, "accuracy": accuracy, "f1 score": f1})

        write_csv(file_name + ".csv", results)

