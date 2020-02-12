import glob
import logging
import os
import warnings

import tifffile

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
