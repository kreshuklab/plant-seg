import glob
import logging
import os

import sys
import warnings

warnings.simplefilter('once', UserWarning)

loggers = {}


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


allowed_data_format = [".tiff", ".tif", ".hdf", ".hdf5", ".h5", ".hd5"]


def load_paths(config):
    if os.path.isdir(config["path"]):
        path = os.path.join(config["path"], "*")
        paths = glob.glob(path)
        only_file = []
        for path in paths:
            _, ext = os.path.splitext(path)
            if os.path.isfile(path) and ext in allowed_data_format:
                print(f" - Valid input file found: {path}")
                only_file.append(path)
            else:
                print(f" - Non-valid input file found: {path}, skipped!")
                warnings.warn(f"Allowed file formats are: {allowed_data_format}")
        return sorted(only_file)

    else:
        path, ext = os.path.splitext(config["path"])
        if ext in allowed_data_format:
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
