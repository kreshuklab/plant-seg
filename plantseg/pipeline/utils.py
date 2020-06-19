import glob
import logging
import os
import warnings
import sys

import tifffile
import h5py
from plantseg.pipeline import gui_logger, H5_KEYS

warnings.simplefilter('once', UserWarning)

loggers = {}

ALLOWED_DATA_FORMAT = [".tiff", ".tif", ".hdf", ".hdf5", ".h5", ".hd5"]


def load_paths(config):
    if os.path.isdir(config["path"]):
        path = os.path.join(config["path"], "*")
        paths = glob.glob(path)
        only_file = []
        for path in paths:
            _, ext = os.path.splitext(path)
            if os.path.isfile(path) and ext in ALLOWED_DATA_FORMAT:
                print(f" - Valid input file found: {path}")
                only_file.append(path)
            else:
                print(f" - Non-valid input file found: {path}, skipped!")
                warnings.warn(f"Allowed file formats are: {ALLOWED_DATA_FORMAT}")
        return sorted(only_file)

    else:
        path, ext = os.path.splitext(config["path"])
        if ext in ALLOWED_DATA_FORMAT:
            return [config["path"]]
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
    # if only one dataset in h5_file return it, otherwise return first from H5_KEYS
    found_keys = list(h5_file.keys())
    if not found_keys:
        raise RuntimeError(f"No datasets found in '{h5_file.filename}'")

    if len(found_keys) == 1:
        return found_keys[0]
    else:
        for h5_key in H5_KEYS:
            if h5_key in found_keys:
                return h5_key

        raise RuntimeError(f"Ambiguous datasets '{found_keys}' in {h5_file.filename}")


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

def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger