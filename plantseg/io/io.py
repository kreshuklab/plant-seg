import os
import warnings

import h5py
import tifffile

TIFF_EXTENSIONS = [".tiff", ".tif"]
H5_EXTENSIONS = [".hdf", ".h5", ".hd5", "hdf5"]

# allowed h5 keys
H5_KEYS = ["raw", "predictions", "segmentation"]
allowed_data_format = TIFF_EXTENSIONS + H5_EXTENSIONS


def read_tiff_voxel_size(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
            voxel_size_unit = image_metadata.get('unit', 'um')
        else:
            # default voxel size
            z = 1.
            voxel_size_unit = 'um'

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x], voxel_size_unit


def read_h5_voxel_size(f, h5key):
    ds = f[h5key]

    # parse voxel_size
    if 'element_size_um' in ds.attrs:
        voxel_size = ds.attrs['element_size_um']
    else:
        warnings.warn('Voxel size not found, returning default [1.0, 1.0. 1.0]', RuntimeWarning)
        voxel_size = [1.0, 1.0, 1.0]

    return voxel_size


def _find_input_key(h5_file):
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


def load_h5(path, key, slices=None, info_only=False):
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


def load_tiff(path, info_only=False):
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
    _, ext = os.path.splitext(path)
    if ext in H5_EXTENSIONS:
        return load_h5(path, key, info_only=info_only)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path, info_only=info_only)

    else:
        print(f"No default found for {ext}, reverting to default loader")
        return default(path)


def load_shape(path, key=None):
    _, data_shape, _, _ = smart_load(path, key=key, info_only=True)
    return data_shape


def create_h5(path, stack, key, voxel_size=(1.0, 1.0, 1.0), mode='a'):
    with h5py.File(path, mode) as f:
        f.create_dataset(key, data=stack, compression='gzip')
        # save voxel_size
        f[key].attrs['element_size_um'] = voxel_size


def list_keys(path, mode='r'):
    with h5py.File(path, mode) as f:
        return [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]


def del_h5_key(path, key, mode='a'):
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
            f.close()


def rename_h5_key(path, old_key, new_key, mode='r+'):
    """ Rename the 'old_key' dataset to 'new_key' """
    with h5py.File(path, mode) as f:
        if old_key in f:
            f[new_key] = f[old_key]
            del f[old_key]
            f.close()


def create_tiff(path, stack, voxel_size, voxel_size_unit='um'):
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
