import os
import warnings
from xml.etree import cElementTree as ElementTree
from typing import Optional, Union
import h5py
import numpy as np
import tifffile

TIFF_EXTENSIONS = [".tiff", ".tif"]
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]

# allowed h5 keys
H5_KEYS = ["raw", "predictions", "segmentation"]
allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS


def _read_imagej_meta(tiff) -> tuple[list[float, float, float], str]:
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    :returns the voxels size and the voxel units
    """
    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    image_metadata = tiff.imagej_metadata
    z = image_metadata.get('spacing', 1.)
    voxel_size_unit = image_metadata.get('unit', 'um')

    tags = tiff.pages[0].tags
    # parse X, Y resolution
    y = _xy_voxel_size(tags, 'YResolution')
    x = _xy_voxel_size(tags, 'XResolution')
    # return voxel size
    return [z, y, x], voxel_size_unit


def _read_ome_meta(tiff) -> tuple[list[float, float, float], str]:
    """
    :returns the voxels size and the voxel units
    """
    xml_om = tiff.ome_metadata
    tree = ElementTree.fromstring(xml_om)

    image_element = [image for image in tree if image.tag.find('Image') != -1]
    if image_element:
        image_element = image_element[0]
    else:
        warnings.warn(f'Error parsing omero tiff meta Image. '
                      f'Reverting to default voxel size (1., 1., 1.) um')
        return [1., 1., 1.], 'um'

    pixels_element = [pixels for pixels in image_element if pixels.tag.find('Pixels') != -1]
    if pixels_element:
        pixels_element = pixels_element[0]
    else:
        warnings.warn(f'Error parsing omero tiff meta Pixels. '
                      f'Reverting to default voxel size (1., 1., 1.) um')
        return [1., 1., 1.], 'um'

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
        if not np.alltrue([_value == units[0] for _value in units]):
            warnings.warn(f'Units are not homogeneous: {units}')

    if x is None:
        x = 1.
        warnings.warn(f'Error parsing omero tiff meta. '
                      f'Reverting to default voxel size x = 1.')

    if y is None:
        y = 1.
        warnings.warn(f'Error parsing omero tiff meta. '
                      f'Reverting to default voxel size y = 1.')

    if z is None:
        z = 1.
        warnings.warn(f'Error parsing omero tiff meta. '
                      f'Reverting to default voxel size z = 1.')

    return [z, y, x], voxel_size_unit


def read_tiff_voxel_size(file_path: str) -> tuple[list[float, float, float], str]:
    """
    :returns the voxels size and the voxel units for imagej and ome style tiff (if absent returns [1, 1, 1], um)
    """
    with tifffile.TiffFile(file_path) as tiff:
        if tiff.imagej_metadata is not None:
            [z, y, x], voxel_size_unit = _read_imagej_meta(tiff)

        elif tiff.ome_metadata is not None:
            [z, y, x], voxel_size_unit = _read_ome_meta(tiff)

        else:
            # default voxel size
            warnings.warn(f'No metadata found. '
                          f'Reverting to default voxel size (1., 1., 1.) um')
            x, y, z = 1., 1., 1.
            voxel_size_unit = 'um'

        return [z, y, x], voxel_size_unit


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


def load_h5(path: str,
            key: str,
            slices: Optional[slice] = None,
            info_only: bool = False) -> Union[tuple, tuple[np.array, tuple]]:
    """
    Load a dataset from a h5 file and returns some meta info about it.
    :param path: Path to the h5file
    :param key: internal key of the desired dataset
    :param slices: Optional, slice to load
    :param info_only: if true will return a tuple with infos such as voxel resolution, units and shape.
    :return: dataset as numpy array and infos
    """
    with h5py.File(path, 'r') as f:
        if key is None:
            key = _find_input_key(f)

        voxel_size = read_h5_voxel_size(f, key)
        file_shape = f[key].shape

        infos = (voxel_size, file_shape, key, 'um')
        if info_only:
            return infos

        file = f[key][...] if slices is None else f[key][slices]

    return file, infos


def load_tiff(path: str, info_only: bool = False) -> Union[tuple, tuple[np.array, tuple]]:
    """
    Load a dataset from a tiff file and returns some meta info about it.
    :param path: path to the tiff files
    :param info_only: if true will return a tuple with infos such as voxel resolution, units and shape.
    :return: dataset as numpy array and infos
    """
    file = tifffile.imread(path)
    try:
        voxel_size, voxel_size_unit = read_tiff_voxel_size(path)
    except:
        # ZeroDivisionError could happen while reading the voxel size
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]
        voxel_size_unit = 'um'

    infos = (voxel_size, file.shape, None, voxel_size_unit)
    if info_only:
        return infos
    else:
        return file, infos


def smart_load(path, key=None, info_only=False, default=load_tiff):
    """
    Smarth load tries to load a file that can be either a h5 or a tiff
    :param path: Path to the h5file
    :param key: internal key of the desired dataset
    :param info_only: if true will return a tuple with infos such as voxel resolution, units and shape.
    :param default: default loader if the type is not understood
    :return:
    """
    _, ext = os.path.splitext(path)
    if ext in H5_EXTENSIONS:
        return load_h5(path, key, info_only=info_only)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path, info_only=info_only)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)


def load_shape(path: str, key: str = None) -> tuple[int, ...]:
    """
    load only the stack shape from a file
    """
    _, data_shape, _, _ = smart_load(path, key=key, info_only=True)
    return data_shape


def create_h5(path: str,
              stack: np.array,
              key: str,
              voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
              mode: str = 'a') -> None:
    """
    Helper function to create a dataset inside a h5 file
    :param path: file path
    :param stack: numpy array
    :param key: key of the dataset
    :param voxel_size: voxel size in micrometers
    :param mode: file io mode ['w', 'a']
    :return:
    """

    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def list_keys(path: str) -> list[str]:
    """
    returns all datasets in a h5 file
    """
    with h5py.File(path, 'r') as f:
        return [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]


def del_h5_key(path: str, key: str, mode: str = 'a') -> None:
    """
    helper function to delete a dataset from a h5file
    """
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path: str, old_key: str, new_key: str, mode='r+') -> None:
    """ Rename the 'old_key' dataset to 'new_key' """
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()


def create_tiff(path: str, stack: np.array, voxel_size: list[float, float, float], voxel_size_unit: str = 'um') -> None:
    """
    Helper function to create a tiff file from a numpy array
    :param path: path of the new file
    :param stack: numpy array
    :param voxel_size: Optional voxel size
    :param voxel_size_unit: Optional units of the voxel size
    :return:
    """
    # taken from: https://pypi.org/project/tifffile docs
    z, y, x = stack.shape
    stack.shape = 1, z, 1, y, x, 1  # dimensions in TZCYXS order
    spacing, y, x = voxel_size
    resolution = (1. / x, 1. / y)
    # Save output results as tiff
    tifffile.imwrite(path,
                     data=stack,
                     dtype=stack.dtype,
                     imagej=True,
                     resolution=resolution,
                     metadata={'axes': 'TZCYXS', 'spacing': spacing, 'unit': voxel_size_unit})
