import warnings
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

# allowed h5 keys
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]
H5_KEYS = ["raw", "predictions", "segmentation"]


def read_h5_voxel_size(f, h5key: str) -> list[float, float, float]:
    """
    :returns the voxels size stored in a h5 dataset (if absent returns [1, 1, 1])
    """
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        voxel_size = ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return voxel_size


def _find_input_key(h5_file) -> str:
    f"""
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

        raise RuntimeError(f"Ambiguous datasets '{found_datasets}' in {h5_file.filename}. "
                           f"plantseg expects only one dataset to be present in input H5.")


def load_h5(path: Union[str, Path],
            key: str,
            slices: Optional[slice] = None,

            info_only: bool = False) -> Union[tuple, tuple[np.array, tuple]]:
    """
    Load a dataset from a h5 file and returns some meta info about it.
    Args:
        path (str): Path to the h5file
        key (str): internal key of the desired dataset
        slices (Optional[slice], optional): Optional, slice to load. Defaults to None.
        info_only (bool, optional): if true will return a tuple with infos such as voxel resolution, units and shape. \
        Defaults to False.

    Returns:
        Union[tuple, tuple[np.array, tuple]]: dataset as numpy array and infos
    """
    with h5py.File(path, mode='r') as f:
        if key is None:
            key = _find_input_key(f)

        voxel_size = read_h5_voxel_size(f, key)
        file_shape = f[key].shape

        infos = (voxel_size, file_shape, key, 'um')
        if info_only:
            return infos

        file = f[key][...] if slices is None else f[key][slices]

    return file, infos


def create_h5(path: Union[str, Path],
              stack: np.array,
              key: str,
              voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
              mode: str = 'a') -> None:
    """
    Helper function to create a dataset inside a h5 file
    Args:
        path (str): file path
        stack (np.array): numpy array to save as dataset in the h5 file
        key (str): key of the dataset in the h5 file
        voxel_size (tuple[float, float, float]: voxel size in micrometers
        mode (str): mode to open the h5 file ['w', 'a']

    Returns:
        None
    """

    with h5py.File(path, mode=mode) as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def write_attribute_h5(path: Union[str, Path], atr_dict: dict, key: str = None) -> None:
    """
    Helper function to add attributes to a h5 file
    Args:
        path (str): file path
        atr_dict (dict): dictionary of attributes to add
        key (str): key of the dataset in the h5 file

    Returns:
        None
    """
    assert Path(path).suffix in H5_EXTENSIONS, f"File {path} is not a h5 file"
    assert Path(path).exists(), f"File {path} does not exist"
    assert isinstance(atr_dict, dict), "atr_dict must be a dictionary"
    assert isinstance(key, str) or key is None, "key must be a string or None"

    with h5py.File(path, mode='r+') as f:
        if key is None:
            file = f
        elif key in f:
            file = f[key]
        else:
            raise KeyError(f"Key {key} not found in {path}")

        for k, v in atr_dict.items():
            if v is None:
                v = 'none'
            file.attrs[k] = v


def read_attribute_h5(path: Union[str, Path], key: str = None) -> dict:
    """
    Helper function to read attributes from a h5 file
    Args:
        path (str): file path
        key (str): key of the dataset in the h5 file

    Returns:
        dict: dictionary of attributes
    """
    assert Path(path).suffix in H5_EXTENSIONS, f"File {path} is not a h5 file"
    assert Path(path).exists(), f"File {path} does not exist"
    assert isinstance(key, str) or key is None, "key must be a string or None"
    with h5py.File(path, mode='r') as f:
        if key is None:
            attrs = f.attrs
        elif key in f:
            attrs = f[key].attrs
        else:
            raise KeyError(f"Key {key} not found in {path}")

        attrs_dict = {}
        for k, v in attrs.items():
            if isinstance(v, str) and v == 'none':
                v = None
            attrs_dict[k] = v
        return attrs_dict


def list_keys(path: Union[str, Path]) -> list[str]:
    """
    List all keys in a h5 file
    Args:
        path: path to the h5 file

    Returns:
        list of keys
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

    with h5py.File(path, mode='r') as h5_f:
        return _recursive_find_keys(h5_f)


def del_h5_key(path: Union[str, Path], key: str, mode: str = 'a') -> None:
    """
    helper function to delete a dataset from a h5file
    """
    with h5py.File(path, mode=mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path: Union[str, Path], old_key: str, new_key: str, mode='r+') -> None:
    """ Rename the 'old_key' dataset to 'new_key' """
    with h5py.File(path, mode=mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()
