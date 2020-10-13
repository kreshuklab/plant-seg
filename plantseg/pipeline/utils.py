import glob
import logging
import os
import warnings

import tifffile
import h5py
from plantseg.pipeline import gui_logger, H5_KEYS

warnings.simplefilter('once', UserWarning)

loggers = {}

ALLOWED_DATA_FORMAT = [".tiff", ".tif", ".hdf", ".hdf5", ".h5", ".hd5"]


def load_paths(base_path):
    assert os.path.exists(base_path), f'File not found: {base_path}'

    if os.path.isdir(base_path):
        path = os.path.join(base_path, "*")
        paths = glob.glob(path)
        only_file = []
        for path in paths:
            _, ext = os.path.splitext(path)
            if os.path.isfile(path) and ext in ALLOWED_DATA_FORMAT:
                only_file.append(path)
        return sorted(only_file)
    else:
        path, ext = os.path.splitext(base_path)
        if ext in ALLOWED_DATA_FORMAT:
            return [base_path]
        else:
            raise RuntimeError(f"Unsupported file type: '{ext}'")


# Copied from https://github.com/beenje/tkinter-logging-text-widget/blob/master/main.py
class QueueHandler(logging.Handler):
    """Class to send logging records to a queue
    It can be used from different threads
    The ConsoleUi class polls this queue to display records in a ScrolledText widget
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


def find_input_key(h5_file):
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
        else:
            # default voxel size
            z = 1.

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size
        return [z, y, x]


def read_h5_voxel_size(file_path):
    with h5py.File(file_path, "r") as f:
        h5key = find_input_key(f)
        ds = f[h5key]

        # parse voxel_size
        if 'element_size_um' in ds.attrs:
            return ds.attrs['element_size_um']
        else:
            gui_logger.warn(f"Cannot find 'element_size_um' attribute for dataset '{h5key}'. "
                            f"Using default voxel_size: {[1., 1., 1.]}")
            return [1., 1., 1.]
