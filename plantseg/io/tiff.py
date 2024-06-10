import warnings
from typing import Union
from xml.etree import cElementTree as ElementTree

import numpy as np
import tifffile

TIFF_EXTENSIONS = [".tiff", ".tif"]


def _read_imagej_meta(tiff) -> tuple[tuple[float, float, float], str]:
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
        return 1.0

    image_metadata = tiff.imagej_metadata
    z = image_metadata.get('spacing', 1.0)
    voxel_size_unit = image_metadata.get('unit', 'um')

    tags = tiff.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
    return (z, y, x), voxel_size_unit


def _read_ome_meta(tiff) -> tuple[tuple[float, float, float], str]:
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
        return (1.0, 1.0, 1.0), 'um'

    pixels_element = [pixels for pixels in image_element if pixels.tag.find('Pixels') != -1]
    if pixels_element:
        pixels_element = pixels_element[0]
    else:
        warnings.warn('Error parsing omero tiff meta Pixels. Reverting to default voxel size (1., 1., 1.) um')
        return (1.0, 1.0, 1.0), 'um'

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

    if x is None:
        x = 1.0
        warnings.warn('Error parsing omero tiff meta. Reverting to default voxel size x = 1.')

    if y is None:
        y = 1.0
        warnings.warn('Error parsing omero tiff meta. Reverting to default voxel size y = 1.')

    if z is None:
        z = 1.0
        warnings.warn('Error parsing omero tiff meta. Reverting to default voxel size z = 1.')

    return (z, y, x), voxel_size_unit


def read_tiff_voxel_size(file_path: str) -> tuple[tuple[float, float, float], str]:
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
            [z, y, x], voxel_size_unit = _read_imagej_meta(tiff)

        elif tiff.ome_metadata is not None:
            [z, y, x], voxel_size_unit = _read_ome_meta(tiff)

        else:
            # default voxel size
            warnings.warn('No metadata found. Reverting to default voxel size (1., 1., 1.) um')
            x, y, z = 1.0, 1.0, 1.0
            voxel_size_unit = 'um'

        return (z, y, x), voxel_size_unit


def load_tiff(path: str, info_only: bool = False) -> Union[tuple, tuple[np.ndarray, tuple]]:
    """
    Load a dataset from a tiff file and returns some meta info about it.
    Args:
        path (str): path to the tiff files to load
        info_only (bool): if true will return a tuple with infos such as voxel resolution, units and shape.

    Returns:
        stack (np.ndarray): numpy array with the data
        infos (tuple): tuple with the voxel size, shape, metadata and voxel size unit (if info_only is True)
    """
    file = tifffile.imread(path)
    try:
        voxel_size, voxel_size_unit = read_tiff_voxel_size(path)
    except ZeroDivisionError:
        # ZeroDivisionError could happen while reading the voxel size
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = (1.0, 1.0, 1.0)
        voxel_size_unit = 'um'

    infos = (voxel_size, file.shape, None, voxel_size_unit)
    if info_only:
        return infos

    return file, infos


def create_tiff(
    path: str,
    stack: np.ndarray,
    voxel_size: tuple[float, float, float],
    voxel_size_unit: str = 'um',
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
    spacing, y, x = voxel_size
    resolution = (1.0 / x, 1.0 / y)
    # Save output results as tiff
    tifffile.imwrite(
        path,
        data=stack,
        dtype=stack.dtype,
        imagej=True,
        resolution=resolution,
        metadata={'axes': 'TZCYXS', 'spacing': spacing, 'unit': voxel_size_unit},
        compression='zlib',
    )


