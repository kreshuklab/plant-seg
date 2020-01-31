import glob
import os


def load_paths(config):
    if os.path.isdir(config["path"]):
        path = os.path.join(config["path"], "*")
        paths = glob.glob(path)
        only_file = []
        for path in paths:
            if os.path.isfile(path):
                only_file.append(path)
        return sorted(only_file)

    else:
        path, ext = os.path.splitext(config["path"])
        if ext in [".tiff", ".tif", ".hdf", ".h5", ".hd5"]:
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
