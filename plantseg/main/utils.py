import glob
import os
import warnings

warnings.simplefilter('once', UserWarning)

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
