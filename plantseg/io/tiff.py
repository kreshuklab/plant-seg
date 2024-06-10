import warnings
from typing import Self
from xml.etree import cElementTree as ElementTree

import numpy as np
import tifffile
from plantseg.io.utils import VoxelSize, DataHandler
from pathlib import Path

TIFF_EXTENSIONS = [".tiff", ".tif"]


def _read_imagej_meta(tiff) -> VoxelSize:
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    Returns the voxel size and the voxel units
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return None

    image_metadata = tiff.imagej_metadata
    z = image_metadata.get('spacing', None)
    voxel_size_unit = image_metadata.get('unit', 'um')

    tags = tiff.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
    
    if x is None or y is None or z is None:
        warnings.warn('Error parsing imagej tiff meta.')
        return VoxelSize()
    
    return VoxelSize(voxels_size=(z, y, x), unit=voxel_size_unit)


def _read_ome_meta(tiff) -> VoxelSize:
    """
    Returns the voxels size and the voxel units
    """
    xml_om = tiff.ome_metadata
    tree = ElementTree.fromstring(xml_om)

    image_element = [image for image in tree if image.tag.find('Image') != -1]
    if image_element:
        image_element = image_element[0]
    else:
        warnings.warn('Error parsing omero tiff meta Image. Reverting to default voxel size (1., 1., 1.) um')
        return VoxelSize()

    pixels_element = [pixels for pixels in image_element if pixels.tag.find('Pixels') != -1]
    if pixels_element:
        pixels_element = pixels_element[0]
    else:
        warnings.warn('Error parsing omero tiff meta Pixels. Reverting to default voxel size (1., 1., 1.) um')
        return VoxelSize()

    units = []
    x, y, z, voxel_size_unit = None, None, None, 'um'

    for key, value in pixels_element.items():
        if key == 'PhysicalSizeX':
            x = float(value)

        elif key == 'PhysicalSizeY':
            y = float(value)

        elif key == 'PhysicalSizeZ':
            z = float(value)

        if key in ['PhysicalSizeXUnit', 'PhysicalSizeYUnit', 'PhysicalSizeZUnit']:
            units.append(value)

    if units:
        voxel_size_unit = units[0]
        if not all(unit == voxel_size_unit for unit in units):
            warnings.warn('Units are not homogeneous: {units}')

    if x is None or y is None or z is None:
        warnings.warn('Error parsing omero tiff meta. ')
        return VoxelSize()

    return VoxelSize(voxels_size=(z, y, x), unit=voxel_size_unit)


def read_tiff_voxel_size(file_path: str) -> VoxelSize:
    """
    Returns the voxels size and the voxel units for imagej and ome style tiff (if absent returns [1, 1, 1], um)

    Args:
        file_path (str): path to the tiff file

    Returns:
        voxel size
        voxel size unit

    """
    with tifffile.TiffFile(file_path) as tiff:
        if tiff.imagej_metadata is not None:
            return _read_imagej_meta(tiff)

        elif tiff.ome_metadata is not None:
            return _read_ome_meta(tiff)

        warnings.warn('No metadata found.')
        return VoxelSize()


def load_tiff(path: str) -> np.ndarray:
    """
    Load a dataset from a tiff file and returns some meta info about it.
    Args:
        path (str): path to the tiff files to load
        info_only (bool): if true will return a tuple with infos such as voxel resolution, units and shape.

    Returns:
        stack (np.ndarray): numpy array with the data
        infos (tuple): tuple with the voxel size, shape, metadata and voxel size unit (if info_only is True)
    """
    return tifffile.imread(path)
    

def create_tiff(
    path: str,
    stack: np.ndarray,
    voxel_size: VoxelSize,
) -> None:
    """
    Create a tiff file from a numpy array

    Args:
        path (str): path of the new file
        stack (np.ndarray): numpy array to save as tiff
        voxel_size (list or tuple): tuple of the voxel size
        voxel_size_unit (str): units of the voxel size

    """
    # taken from: https://pypi.org/project/tifffile docs
    z, y, x = stack.shape
    stack = stack.reshape(1, z, 1, y, x, 1)  # dimensions in TZCYXS order
    spacing, y, x = VoxelSize.voxels_size
    resolution = (1.0 / x, 1.0 / y)
    # Save output results as tiff
    tifffile.imwrite(
        path,
        data=stack,
        dtype=stack.dtype,
        imagej=True,
        resolution=resolution,
        metadata={'axes': 'TZCYXS', 'spacing': spacing, 'unit': voxel_size.unit},
        compression='zlib',
    )



class TiffDataHandler(DataHandler):
    """
    Class to handle data loading, and metadata retrieval from a Zarr file.

    Attributes:
        path (Path): path to the zarr file
        key (str): key of the dataset in the zarr file
    """
    _data: np.ndarray = None
    _voxel_size = None

    def __init__(self, path: Path):
        self.path = path
        
    def __repr__(self):
        return f"TiffDataHandler(path={self.path})"
    
    @classmethod
    def from_data_handler(cls, data_handler: DataHandler, path: Path) -> Self:
        """
        Create a TiffDataHandler object from a DataHandler object.
        
        Args:
            data_handler (DataHandler): DataHandler object
            
        Returns:
            TiffDataHandler: TiffDataHandler object
        """
        zarr_handler = cls(path)
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
        
        self._data = load_tiff(self.path)
        return self._data
        
    def write_data(self, **kwargs) -> None:
        """
        Write the dataset to the h5 file.
        """
        create_tiff(path=self.path, stack=self._data, voxel_size=self._voxel_size)
        
    def get_shape(self) -> tuple[int]:
        """
        Get the shape of the dataset.
        """
        return self.get_data().shape
        
    
    def get_voxel_size(self) -> VoxelSize:
        """
        Get the voxel size of the dataset.
        """
        if self._voxel_size is not None:
            return self._voxel_size

        return self._voxel_size