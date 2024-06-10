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

import warnings
from typing import Optional, Self
from plantseg.io.utils import VoxelSize, DataHandler


ZARR_EXTENSIONS = [".zarr"]
ZARR_KEYS = ["raw", "predictions", "segmentation"]


def _validate_zarr_file(path: Path) -> None:
    assert path.suffix in ZARR_EXTENSIONS, f"File extension not supported. Supported extensions: {ZARR_EXTENSIONS}"
    assert path.exists(), f"File not found: {path}"


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
        
def _get_zarr_dataset(zarr_path: Path, key: Optional[str] = None) -> zarr.Array:
    zarr_file = zarr.open_group(zarr_path, mode='r')
    if key is None:
        key = _find_input_key(zarr_file)

    ds = zarr_file[key]
    if not isinstance(ds, zarr.Array):
        raise ValueError(f"'{key}' is not a zarr.Array.")
    
    return zarr.open_array(zarr_path, mode='r', path=key)


def load_zarr(
    path: Path,
    key: Optional[str],
    slices: Optional[slice] = None,
) -> Union[
    tuple[tuple[float, float, float], Any, str, str],
    tuple[np.ndarray, tuple[tuple[float, float, float], Any, str, str]],
]:
    """Load a dataset from a Zarr file and return it or its meta-information.

    Args:
        path (Path): The path to the Zarr file.
        key (str): The internal key of the desired dataset.
        slices (Optional[slice], optional): Slice to load. Defaults to None.

    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: The dataset as a NumPy array, and meta-information.
    """
    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    data = data[...] if slices is None else data[slices]
    return data


def read_zarr_shape(path: Path, key: Optional[str] = None) -> tuple[int]:
    """Read the shape of a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (Optional[str], optional): The internal key of the desired dataset. Defaults to None.

    Returns:
        tuple[int]: The shape of the dataset.
    """
    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    return data.shape


def read_zarr_voxel_size(path: Path, key: str) -> VoxelSize:
    """Read the voxel size of a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str): The internal key of the desired dataset.

    Returns:
        VoxelSize: The voxel size of the dataset.
    """
    _validate_zarr_file(path)
    data = _get_zarr_dataset(path, key)
    
    if 'element_size_um' in data.attrs:
        return VoxelSize(voxels_size=data.attrs['element_size_um'])
    elif 'resolution' in data.attrs:
        return VoxelSize(voxels_size=data.attrs['resolution'])
    
    warnings.warn(f"Voxel size not found in {path}.")
    return VoxelSize()


def create_zarr(
    path: Path,
    stack: np.ndarray,
    key: str,
    voxel_size: VoxelSize,
    mode: str = 'a',
) -> None:
    """
    Create a Zarr array from a NumPy array.

    Args:
        path (Path): The path to the Zarr file.
        stack (np.ndarray): The NumPy array to save as a dataset.
        key (str): The internal key of the desired dataset.
        voxel_size (VoxelSize): The voxel size of the dataset.
        mode (str): The mode to open the Zarr file ['w', 'a'].

    """
    zarr_file = zarr.open_group(path, mode)
    zarr_file.create_dataset(key, data=stack, compression='gzip', overwrite=True)
    zarr_file[key].attrs['element_size_um'] = voxel_size.voxels_size


def list_keys(path: Path) -> list[str]:
    """
    List all keys in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.

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


def del_zarr_key(path: Path, key: str, mode: str = 'a') -> None:
    """
    Delete a dataset from a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        key (str): The internal key of the dataset to be deleted.
        mode (str): The mode to open the Zarr file ['w', 'a'].

    """
    zarr_file = zarr.open_group(path, mode)
    if key in zarr_file:
        del zarr_file[key]


def rename_zarr_key(path: Path, old_key: str, new_key: str, mode='r+') -> None:
    """
    Rename a dataset in a Zarr file.

    Args:
        path (Path): The path to the Zarr file.
        old_key (str): The current key of the dataset.
        new_key (str): The new key for the dataset.
        mode (str): The mode to open the Zarr file ['r+'].

    """
    zarr_file = zarr.open_group(path, mode)
    if old_key in zarr_file:
        zarr_file[new_key] = zarr_file[old_key]
        del zarr_file[old_key]


class ZarrDataHandler:
    """
    Class to handle data loading, and metadata retrieval from a Zarr file.

    Attributes:
        path (Path): path to the zarr file
        key (str): key of the dataset in the zarr file
    """
    _data: np.ndarray = None
    _voxel_size = None

    def __init__(self, path: Path, key: str):
        self.path = path
        self.key = key
        
    def __repr__(self):
        return f"ZarrDataHandler(path={self.path}, key={self.key})"
    
    @classmethod
    def from_data_handler(cls, data_handler: DataHandler, path: Path, key: str) -> Self:
        """
        Create a ZarrDataHandler object from a DataHandler object.
        
        Args:
            data_handler (DataHandler): DataHandler object
            
        Returns:
            ZarrDataHandler: ZarrDataHandler object
        """
        zarr_handler = cls(path, key)
        zarr_handler._data = data_handler.get_data()
        zarr_handler._voxel_size = data_handler.get_voxel_size()
        

    def get_data(self, slices=None) -> np.ndarray:
        """
        Load the dataset from the h5 file.

        Returns:
            np.ndarray: dataset as numpy array
        """
        if self._data is not None:
            return self._data
        
        self._data = load_zarr(self.path, self.key, slices)
        return self._data
        
    def write_data(self, **kwargs) -> None:
        """
        Write the dataset to the h5 file.
        """
        create_zarr(path=self.path, stack=self._data, key=self.key, **kwargs)
        
    def get_shape(self) -> tuple[int]:
        """
        Get the shape of the dataset.
        """
        if self._data is not None:
            return self._data.shape
        
        return read_zarr_shape(self.path, self.key)
        
    
    def get_voxel_size(self) -> VoxelSize:
        """
        Get the voxel size of the dataset.
        """
        if self._voxel_size is not None:
            return self._voxel_size
            
        return read_zarr_voxel_size(self.path, self.key)