import os

import numpy as np
from typing import Union
from plantseg.io.h5 import load_h5, H5_EXTENSIONS
from plantseg.io.tiff import load_tiff, TIFF_EXTENSIONS
from plantseg.io.pil import load_pill, PIL_EXTENSIONS
from plantseg.io.zarr import load_zarr, ZARR_EXTENSIONS

allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS + PIL_EXTENSIONS + ZARR_EXTENSIONS


def smart_load(path, key=None, info_only=False, default=load_tiff) -> Union[tuple, tuple[np.array, tuple]]:
    """
    Smart load tries to load a file that can be either a h5 or a tiff file
    Args:
        path (str): path to the file to load
        key (str): key of the dataset to load (if h5)
        info_only (bool): if true will return a tuple with infos such as voxel resolution, units and shape.
        default (callable): default loader if the type is not understood (default: load_tiff)

    Returns:
        Union[tuple, tuple[np.array, tuple]]: loaded data as numpy array and infos

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
    load only the stack shape from a file
    """
    _, data_shape, _, _ = smart_load(path, key=key, info_only=True)
    return data_shape
