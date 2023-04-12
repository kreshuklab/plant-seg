import collections
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.slice_builder import SliceBuilder


class ArrayDataset(Dataset):
    """
    Based on pytorch-3dunet  AbstractHDF5Dataset
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py

    Inference only implementation of torch.utils.data.Dataset
    """

    def __init__(self, raw: np.ndarray, slice_builder: SliceBuilder, augs: Callable[[np.ndarray], torch.Tensor],
                 verbose_logging: bool = True):
        """
        Args:
            raw (np.ndarray): raw data
            slice_builder (SliceBuilder): slice builder
            augs (Callable): data augmentation pipeline
            verbose_logging (bool): if True, log info messages
        """
        self.raw = raw
        self.augs = augs
        self.raw_slices = slice_builder.raw_slices

        if verbose_logging:
            gui_logger.info(f'Number of patches: {len(self.raw_slices)}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice and augment
        raw_patch_transformed = self.augs(self.raw[raw_idx])
        # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
        return raw_patch_transformed, raw_idx

    def __len__(self):
        return len(self.raw_slices)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
