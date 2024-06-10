from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from pathlib import Path
from typing import Self, Optional
from plantseg.io.utils import VoxelSize, DataHandler


PIL_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def load_pil(path: Path) -> np.ndarray:
    """
    Load a tiff file using PIL

    Args:
        path (Path): path to an image file to load (png, jpg, jpeg).
        info_only (bool, optional): if true will return a tuple with the default infos.
            true infos can not be extracted from a PIL image. Defaults to False.
    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: loaded data as numpy array and infos
    """

    image = Image.open(path)
    if image.mode != "L":
        image = grayscale(image)

    image = np.array(image)

    return image


class PilDataHandler:
    """
    Class to handle data loading, and metadata retrieval from a Zarr file.

    Attributes:
        path (Path): path to the zarr file
        key (str): key of the dataset in the zarr file
    """

    _data: Optional[np.ndarray] = None
    _voxel_size = VoxelSize()
    key: str = ""

    def __init__(self, path: Path):
        self.path = path

    def __repr__(self):
        return f"PilDataHandler(path={self.path})"

    @classmethod
    def from_data_handler(cls, data_handler: DataHandler, path: Path, key: Optional[str] = None) -> Self:
        raise NotImplementedError("PilDataHandler does not support from_data_handler")

    def get_data(self, slices=None) -> np.ndarray:
        """
        Load the dataset from the h5 file.

        Returns:
            np.ndarray: dataset as numpy array
        """
        if self._data is None:
            self._data = load_pil(self.path)

        return self._data

    def write_data(self, **kwargs) -> None:
        raise NotImplementedError("PilDataHandler does not support write_data")

    def get_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the dataset.
        """
        if self._data is None:
            self._data = load_pil(self.path)

        return self._data.shape

    def get_voxel_size(self) -> VoxelSize:
        """
        Get the voxel size of the dataset.
        """
        return self._voxel_size
