"""
Reading and writing zarrs. Created in the same format as h5.py.

Notes:
1. Although the function is called "open", there is no need to close an array: data is automatically flushed to disk, and files are automatically closed whenever an array is modified. [Ref](https://zarr.readthedocs.io/en/stable/tutorial.html#persistent-arrays).
2. An open file either gives a `zarr.Array` or `zarr.Group` object. In PlantSeg, `zarr.open` is expected to return a `zarr.Group` object.
3. Zarr works with string paths, not `pathlib.Path` objects. No need to upgrade to `pathlib`.
"""

import warnings
from typing import Optional, Union, Any
from pathlib import Path

import zarr
import numpy as np

ZARR_EXTENSIONS = [".zarr"]
ZARR_KEYS = ["raw", "predictions", "segmentation"]


def read_zarr_voxel_size(zarr_file: zarr.Group, zarr_key: str) -> tuple[float, float, float]:
    """Get the voxel size stored in a Zarr dataset.

    Args:
        zarr_file (zarr.Group): The Zarr file (or group) containing the dataset.
        zarr_key (str): The key identifying the desired dataset.

    Returns:
        tuple[float, float, float]: The voxel size in the dataset, or (1.0, 1.0, 1.0) if not found.
    """
    zarr_array = zarr_file[zarr_key]

    if 'element_size_um' in zarr_array.attrs:
        return zarr_array.attrs['element_size_um']
    elif 'resolution' in zarr_array.attrs:
        return zarr_array.attrs['resolution']
    else:
        warnings.warn('Voxel size not found, returning default (1.0, 1.0, 1.0)', RuntimeWarning)
        return (1.0, 1.0, 1.0)


def _find_input_key(zarr_file: zarr.Group) -> str:
    """Return the first matching key in `ZARR_KEYS` or the key to the only dataset if just one is found.

    Args:
        zarr_file (zarr.Group): The Zarr file (or group) containing datasets.

    Returns:
        str: The key of the matching dataset.

    Raises:
        RuntimeError: If no datasets are found or the dataset is ambiguous.
    """
    found_datasets = []

    def visitor_func(name, node):
        if isinstance(node, zarr.Array):
            found_datasets.append(name)

    zarr_file.visititems(visitor_func)

    if not found_datasets:
        raise RuntimeError(f"No datasets found - verify '{zarr_file.tree()}'.")

    if len(found_datasets) == 1:
        return found_datasets[0]
    else:
        for zarr_key in ZARR_KEYS:
            if zarr_key in found_datasets:
                return zarr_key

        raise RuntimeError(
            f"Ambiguous datasets '{found_datasets}' in {zarr_file}. "
            f"PlantSeg expects only one dataset in the input Zarr."
        )


def load_zarr(
    path: str,
    key: Optional[str],
    slices: Optional[slice] = None,
    info_only: bool = False,
) -> Union[
    tuple[tuple[float, float, float], Any, str, str],
    tuple[np.ndarray, tuple[tuple[float, float, float], Any, str, str]],
]:
    """Load a dataset from a Zarr file and return it or its meta-information.

    Args:
        path (str): The path to the Zarr file.
        key (str): The internal key of the desired dataset.
        slices (Optional[slice], optional): Slice to load. Defaults to None.
        info_only (bool, optional): If True, return only the meta-information.

    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: The dataset as a NumPy array, and meta-information.
    """
    zarr_file = zarr.open_group(path, mode='r')

    if key is None:
        key = _find_input_key(zarr_file)

    voxel_size = read_zarr_voxel_size(zarr_file, key)
    file_shape = zarr_file[key].shape

    infos = (voxel_size, file_shape, key, 'um')
    if info_only:
        return infos

    zarr_array = zarr.open_array(path, mode='r', path=key)
    data = zarr_array[...] if slices is None else zarr_array[slices]
    return data, infos


def create_zarr(
    path: str,
    stack: np.ndarray,
    key: str,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = 'a',
) -> None:
    """
    Create a Zarr array from a NumPy array.

    Args:
        path (str): The file path to the Zarr file.
        stack (np.ndarray): The NumPy array to save as a dataset.
        key (str): The internal key of the desired dataset.
        voxel_size (tuple[float, float, float]): The voxel size in micrometers.
        mode (str): The mode to open the Zarr file ['w', 'a'].

    """
    zarr_file = zarr.open_group(path, mode)
    zarr_file.create_dataset(key, data=stack, compression='gzip', overwrite=True)
    zarr_file[key].attrs['element_size_um'] = voxel_size


def list_keys(path: str) -> list[str]:
    """
    List all keys in a Zarr file.

    Args:
        path (str): The path to the Zarr file.

    Returns:
        keys (list[str]): A list of keys in the Zarr file.
    """

    def _recursive_find_keys(zarr_group: zarr.Group, base: Path = Path('')) -> list[str]:
        _list_keys = []
        for key, dataset in zarr_group.items():
            if isinstance(dataset, zarr.Group):
                new_base = base / key
                _list_keys.extend(_recursive_find_keys(dataset, new_base))
            elif isinstance(dataset, zarr.Array):
                _list_keys.append(str(base / key))
        return _list_keys

    zarr_file = zarr.open_group(path, 'r')
    return _recursive_find_keys(zarr_file)


def del_zarr_key(path: str, key: str, mode: str = 'a') -> None:
    """
    Delete a dataset from a Zarr file.

    Args:
        path (str): The path to the Zarr file.
        key (str): The internal key of the dataset to be deleted.
        mode (str): The mode to open the Zarr file ['w', 'a'].

    """
    zarr_file = zarr.open_group(path, mode)
    if key in zarr_file:
        del zarr_file[key]


def rename_zarr_key(path: str, old_key: str, new_key: str, mode='r+') -> None:
    """
    Rename a dataset in a Zarr file.

    Args:
        path (str): The path to the Zarr file.
        old_key (str): The current key of the dataset.
        new_key (str): The new key for the dataset.
        mode (str): The mode to open the Zarr file ['r+'].

    """
    zarr_file = zarr.open_group(path, mode)
    if old_key in zarr_file:
        zarr_file[new_key] = zarr_file[old_key]
        del zarr_file[old_key]
