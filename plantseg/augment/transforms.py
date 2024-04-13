from typing import Optional

import numpy as np
import torch

class Compose:
    """
    Compose several transforms together.

    Args:
        transforms (list of callable): The list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m


class ToTensor:
    """
    Convert a numpy.ndarray to a torch.Tensor.

    Args:
        expand_dims (bool): If True, adds a channel dimension to the input data.
        dtype (torch.dtype, optional): The desired data type of the output tensor. Defaults to torch.float32.

    Raises:
        AssertionError: If the input is not a 3D (DxHxW) or 4D (CxDxHxW) image.
    """

    def __init__(self, expand_dims: bool, dtype=torch.float32):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m: np.ndarray) -> torch.Tensor:
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)
        return torch.from_numpy(m).type(self.dtype)


class Standardize:
    """
    Apply Z-score normalization (0-mean, 1-std) to a given input tensor.

    Args:
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-10.
        mean (Optional[float]): The mean to use for standardization. If None, the mean of the input is used. Defaults to None.
        std (Optional[float]): The standard deviation to use for standardization. If None, the std of the input is used. Defaults to None.
        channelwise (bool, optional): Whether to apply the normalization channel-wise. Defaults to False.

    Raises:
        AssertionError: If mean or std is provided, both must be provided.
    """


    def __init__(self, eps: float = 1e-10, mean: Optional[float] = None, std: Optional[float] = None, channelwise: bool = False):
        if (mean is None) != (std is None):
            raise ValueError("Both 'mean' and 'std' must be provided together or not at all.")

        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.mean is not None and self.std is not None:
            mean, std = self.mean, self.std
        else:
            axes = tuple(range(1, m.ndim)) if self.channelwise else None
            mean = np.mean(m, axis=axes, keepdims=True)
            std = np.std(m, axis=axes, keepdims=True)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


def get_test_augmentations(expand_dims=True) -> Compose:
    """
    Returns a list of data augmentation transforms for inference.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
    """
    return Compose([
        ToTensor(expand_dims=expand_dims)
    ])
