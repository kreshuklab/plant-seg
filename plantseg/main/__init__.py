import glob
import os
import warnings

warnings.simplefilter('once', UserWarning)
allowed_data_format = [".tiff", ".tif", ".hdf", ".h5", ".hd5"]


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
                warnings.warn("Allowed file formats are: .tiff, .tif, .hdf, .h5, .hd5.")
        return sorted(only_file)

    else:
        path, ext = os.path.splitext(config["path"])
        if ext in allowed_data_format:
            return [config["path"]]
        else:
            print("Data extension not understood")
            raise NotImplementedError


class dummy:
    def __init__(self, paths, phase):
        self.phase = phase
        self.paths = paths

    def __call__(self,):
        print(f"Skipping {self.phase}: Nothing to do")
        return self.paths
