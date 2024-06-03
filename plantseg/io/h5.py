import warnings
from typing import Optional, Union

import h5py
import numpy as np

# allowed h5 keys
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]
H5_KEYS = ["raw", "predictions", "segmentation"]


def read_h5_voxel_size(f, h5key: str) -> tuple[float, float, float]:
    """
    returns the voxels size stored in a h5 dataset (if absent returns [1, 1, 1])
    """
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        return ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        return (1.0, 1.0, 1.0)


def _find_input_key(h5_file) -> str:
    """
    returns the first matching key in H5_KEYS or only one dataset is found the key to that dataset
    """
    found_datasets = []

    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            found_datasets.append(name)

    h5_file.visititems(visitor_func)

    if not found_datasets:
        raise RuntimeError(f"No datasets found in '{h5_file.filename}'")

    if len(found_datasets) == 1:
        return found_datasets[0]
    else:
        for h5_key in H5_KEYS:
            if h5_key in found_datasets:
                return h5_key

        raise RuntimeError(
            f"Ambiguous datasets '{found_datasets}' in {h5_file.filename}. "
            f"plantseg expects only one dataset to be present in input H5."
        )


def load_h5(
    path: str,
    key: str,
    slices: Optional[slice] = None,
    info_only: bool = False,
) -> Union[tuple, tuple[np.ndarray, tuple]]:
    """
    Load a dataset from a h5 file and returns some meta info about it.

    Args:
        path (str): Path to the h5file
        key (str): internal key of the desired dataset
        slices (Optional[slice], optional): Optional, slice to load. Defaults to None.
        info_only (bool, optional): if true will return a tuple with infos such as voxel resolution, units and shape. \
        Defaults to False.

    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: dataset as numpy array and infos
    """
    with h5py.File(path, 'r') as f:
        if key is None:
            key = _find_input_key(f)

        voxel_size = read_h5_voxel_size(f, key)

        ds = f[key]
        if not isinstance(ds, h5py.Dataset):
            raise ValueError(f"'{key}' is not a h5py.Dataset.")
        file_shape = ds.shape

        infos = (voxel_size, file_shape, key, 'um')
        if info_only:
            return infos

        file = ds[...] if slices is None else ds[slices]

    return file, infos


def create_h5(
    path: str,
    stack: np.ndarray,
    key: str,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mode: str = 'a',
) -> None:
    """
    Create a dataset inside a h5 file from a numpy array.

    Args:
        path (str): file path.
        stack (np.ndarray): numpy array to save as dataset in the h5 file.
        key (str): key of the dataset in the h5 file.
        voxel_size (tuple[float, float, float]: voxel size in micrometers.
        mode (str): mode to open the h5 file ['w', 'a'].

    """

    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def list_keys(path: str) -> list[str]:
    """
    List all keys in a h5 file

    Args:
        path (str): path to the h5 file

    Returns:
        keys (list[str]): A list of keys in the h5 file.

    """

    def _recursive_find_keys(f, base='/'):
        _list_keys = []
        for key, dataset in f.items():
            if isinstance(dataset, h5py.Group):
                new_base = f"{base}{key}/"
                _list_keys += _recursive_find_keys(dataset, new_base)

            elif isinstance(dataset, h5py.Dataset):
                _list_keys.append(f'{base}{key}')
        return _list_keys

    with h5py.File(path, 'r') as h5_f:
        return _recursive_find_keys(h5_f)


def del_h5_key(path: str, key: str, mode: str = 'a') -> None:
    """
    helper function to delete a dataset from a h5file

    Args:
        path (str): path to the h5file
        key (str): key of the dataset to delete
        mode (str): mode to open the h5 file ['r', 'r+']

    """
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path: str, old_key: str, new_key: str, mode='r+') -> None:
    """
    Rename the 'old_key' dataset to 'new_key'

    Args:
        path (str): path to the h5 file
        old_key (str): old key name
        new_key (str): new key name
        mode (str): mode to open the h5 file ['r', 'r+']

    """
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()
