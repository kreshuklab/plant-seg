import logging

import h5py
import numpy as np
from torch.utils.data import Dataset

from plantseg.prediction.utils.slice_builder import FilterSliceBuilder

logger = logging.getLogger(__name__)


# copied from https://github.com/wolny/pytorch-3dunet
class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        augmenter (transforms.Augmenter): list of augmentations to be applied to the raw and label data sets
        patch_shape (tuple): shape of the patch to be extracted from the raw data set
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
        self,
        file_path,
        augmenter,
        patch_shape,
        raw_internal_path="raw",
        label_internal_path="label",
        global_normalization=True,
    ):
        self.file_path = file_path

        with h5py.File(file_path, "r") as f:
            self.raw = self.load_dataset(f, raw_internal_path)
            stats = calculate_stats(self.raw, global_normalization)
            self.augmenter = augmenter
            self.raw_transform = self.augmenter.raw_transform(stats)

            # create label/weight transform only in train/val phase
            self.label_transform = self.augmenter.label_transform()
            self.label = self.load_dataset(f, label_internal_path)
            self._check_volume_sizes(self.raw, self.label)

            # build slice indices for raw and label data sets
            slice_builder = FilterSliceBuilder(
                self.raw,
                self.label,
                patch_shape=patch_shape,
                stride_shape=tuple(i // 2 for i in patch_shape),
            )
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices

            self.patch_count = len(self.raw_slices)
            logger.info(f"{self.patch_count} patches found in {file_path}")

    @staticmethod
    def load_dataset(input_file, internal_path):
        ds = input_file[internal_path][:]
        assert ds.ndim in [
            3,
            4,
        ], (
            f"Invalid dataset dimension: {ds.ndim}. Supported dataset formats: (C, Z, Y, X) or (Z, Y, X)"
        )
        return ds

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        # get the slice for a given index 'idx'
        label_idx = self.label_slices[idx]
        label_patch_transformed = self.label_transform(self.label[label_idx])
        # return the transformed raw and label patches
        return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        raise NotImplementedError

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
        assert label.ndim in [3, 4], "Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)"

        assert _volume_shape(raw) == _volume_shape(label), (
            "Raw and labels have to be of the same size"
        )


def calculate_stats(images, global_normalization=True):
    """
    Calculates min, max, mean, std given a list of nd-arrays
    """
    if global_normalization:
        # flatten first since the images might not be the same size
        flat = np.concatenate([img.ravel() for img in images])
        pmin, pmax, mean, std = (
            np.percentile(flat, 1),
            np.percentile(flat, 99.6),
            np.mean(flat),
            np.std(flat),
        )
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {"pmin": pmin, "pmax": pmax, "mean": mean, "std": std}
