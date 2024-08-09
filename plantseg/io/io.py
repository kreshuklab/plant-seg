import os

import numpy as np
from pathlib import Path
from typing import Optional
from plantseg.io.h5 import load_h5, H5_EXTENSIONS
from plantseg.io.tiff import load_tiff, TIFF_EXTENSIONS
from plantseg.io.pil import load_pil, PIL_EXTENSIONS
from plantseg.io.zarr import load_zarr, ZARR_EXTENSIONS

allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS + PIL_EXTENSIONS + ZARR_EXTENSIONS


def smart_load(path: Path, key: Optional[str] = None, default=load_tiff) -> np.ndarray:
    """
    Load a dataset from a file and returns some meta info about it. The loader is chosen based on the file extension.
    Supported formats are: tiff, h5, zarr, and PIL images.
    If the format is not supported, a default loader can be provided (default: load_tiff).

    Args:
        path (Path): path to the file to load.
        key (str): key of the dataset to load (if h5 or zarr).
        default (callable): default loader if the type is not understood.

    Returns:
        stack (np.ndarray): numpy array with the image data.

    Examples:
        >>> data = smart_load('path/to/file.tif')
        >>> data = smart_load('path/to/file.h5', key='raw')

    """
    _, ext = os.path.splitext(path)
    if ext in H5_EXTENSIONS:
        return load_h5(path, key)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path)

    elif ext in PIL_EXTENSIONS:
        return load_pil(path)

    elif ext in ZARR_EXTENSIONS:
        return load_zarr(path, key)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)
