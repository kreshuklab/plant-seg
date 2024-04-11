import collections
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.slice_builder import SliceBuilder


def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape, before it is split into patches.

    Parameters:
    - image (np.ndarray): The input image array to be padded.
    - padding_shape (tuple of ints): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
    - np.ndarray: The mirror-padded image.
    """
    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]
    return np.pad(image, pad_width, mode='reflect')


def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.
    Parameters:
    - m (np.array): The input array to be unpadded.
    - padding_shape (tuple of ints, optional): The amount of padding to remove from each dimension.
      Assumes the tuple length matches the array dimensions.
    Returns:
    - np.array: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]


class ArrayDataset(Dataset):
    """
    Based on pytorch-3dunet  AbstractHDF5Dataset
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/datasets/hdf5.py

    Inference only implementation of torch.utils.data.Dataset
    """

    def __init__(
        self,
        raw: np.ndarray,
        slice_builder: SliceBuilder,
        augs: Callable[[np.ndarray], torch.Tensor],
        halo_shape: Optional[Tuple[int, int, int]] = None,
        verbose_logging: bool = True,
    ):
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

        if halo_shape is None:
            halo_shape = (0, 0, 0)
        self.halo_shape = halo_shape
        self.raw_padded = mirror_pad(self.raw, self.halo_shape)

        if verbose_logging:
            gui_logger.info(f'Number of patches: {len(self.raw_slices)}')

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        raw_idx_padded = tuple(slice(this_index.start, this_index.stop + 2 * this_halo, None) for this_index, this_halo in zip(raw_idx, self.halo_shape))
        raw_patch = self.raw_padded[raw_idx_padded]
        raw_patch_transformed = self.augs(raw_patch)

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
