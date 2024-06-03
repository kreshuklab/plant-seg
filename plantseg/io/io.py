import os

import numpy as np
from typing import Union
from plantseg.io.h5 import load_h5, H5_EXTENSIONS
from plantseg.io.tiff import load_tiff, TIFF_EXTENSIONS
from plantseg.io.pil import load_pill, PIL_EXTENSIONS
from plantseg.io.zarr import load_zarr, ZARR_EXTENSIONS

allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS + PIL_EXTENSIONS + ZARR_EXTENSIONS


def smart_load(path, key=None, info_only=False, default=load_tiff) -> Union[tuple, tuple[np.ndarray, tuple]]:
    """
    Load a dataset from a file and returns some meta info about it. The loader is chosen based on the file extension.
    Supported formats are: tiff, h5, zarr, and PIL images.
    If the format is not supported, a default loader can be provided (default: load_tiff).

    Args:
        path (str): path to the file to load.
        key (str): key of the dataset to load (if h5).
        info_only (bool): if true will return a tuple with infos such as voxel resolution, units and shape.
        default (callable): default loader if the type is not understood.

    Returns:
        stack (np.ndarray): numpy array with the image data.
        infos (tuple): tuple with the voxel size, shape, metadata and voxel size unit (if info_only is True).

    Examples:
        >>> data, infos = smart_load('path/to/file.tif')
        >>> data, infos = smart_load('path/to/file.h5', key='raw')

    """
    _, ext = os.path.splitext(path)
    if ext in H5_EXTENSIONS:
        return load_h5(path, key, info_only=info_only)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path, info_only=info_only)

    elif ext in PIL_EXTENSIONS:
        return load_pill(path, info_only=info_only)

    elif ext in ZARR_EXTENSIONS:
        return load_zarr(path, key, info_only=info_only)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)


def load_shape(path: str, key: str = None) -> tuple[int, ...]:
    """
    Load only the stack shape from a file using the smart loader.

    Args:
        path (str): path to the file to load.
        key (str): key of the dataset to load (if h5 or zarr).

    Returns:
        shape (tuple[int, ...]) shape of the image stack.

    Examples:
        >>> shape = load_shape('path/to/file.tif')
        >>> print(shape)
        (10, 512, 512)
    """
    _, data_shape, _, _ = smart_load(path, key=key, info_only=True)
    return data_shape
