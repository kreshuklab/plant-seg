import glob
import logging
import os
import warnings

from plantseg.io import allowed_data_format

warnings.simplefilter('once', UserWarning)

loggers = {}

SUPPORTED_TYPES = ["labels", "data_float32", "data_uint8"]


def load_paths(base_path):
    assert os.path.exists(base_path), f'File not found: {base_path}'

    if os.path.isdir(base_path):
        path = os.path.join(base_path, "*")
        paths = glob.glob(path)
        only_file = []
        for path in paths:
            _, ext = os.path.splitext(path)
            if os.path.isfile(path) and ext in allowed_data_format:
                only_file.append(path)
        return sorted(only_file)
    else:
        path, ext = os.path.splitext(base_path)
        if ext in allowed_data_format:
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
