import h5py
import numpy as np
import glob
from scipy.ndimage import zoom
from skimage.morphology import disk
from skimage.filters import median
from vigra.filters import gaussianSmoothing, discMedian
import tifffile
import os


class DataPostProcessing3D:
    def __init__(self, config, paths):
        self.paths = paths

        # convert from tiff
        self.safe = config["safe"]
        self.safe_directory = config["safe_directory"]
        self.convert = config["tiff"]

        # rescaling
        self.factor = config["factor"]
        self.order = config["order"]

    def __call__(self, paths):
        for path in self.paths:
            print(f"Postprocessing {path}")
            with h5py.File(path, "r") as f:
                image = f[self.dataset][...]

            image = image if image.ndim == 3 else image[0]
            image = self.down_sample(image, self.factor, self.order)

            if self.safe:
                os.makedirs(f"{ os.path.dirname(path)}/{ self.safe_directory }/", exist_ok=True)
                file_name = os.path.split(os.path.basename(path))
                h5_file_path = f"{os.path.dirname(path)}/{self.safe_directory}/{file_name}"
            else:
                h5_file_path = path

            image = image.astype(np.uint16)
            if self.convert:
                tiff_file_path = f"{os.path.splitext(h5_file_path)[0]}.tiff"
                tifffile.imsave(tiff_file_path, data=image, dtype=image.dtype,
                                bigtiff=True, imagej=True,
                                resolution=(1, 1), metadata={'spacing': 1, 'unit': 'um'})
            else:
                with h5py.File(h5_file_path, "w") as f:
                    f.create_dataset(self.dataset, data=image, chunks=True,  compression='gzip')

    @staticmethod
    def down_sample(image, factor, order):
        return zoom(image, zoom=factor, order=order)


class DataPreProcessing3D:
    def __init__(self, config, paths):
        self.paths = paths

        # convert from tiff
        self.safe = config["safe"]
        self.safe_directory = config["safe_directory"]
        self.convert = config["tiff"]
        self.dataset = "raw"

        if "filter" in config.keys():
            # filters
            if "median" == config["filter"]:
                self.param = config["param"]
                self.filter = self.median

            elif "gaussian" == config["filter"]:
                self.param = config["param"]
                self.filter = self.gaussian
            else:
                raise NotImplementedError
        else:
            self.param = 0
            self.filter = self.dummy

        # rescaling
        self.factor = config["factor"]
        self.order = config["order"]

    def __call__(self,):
        for path in self.paths:
            print(f"Preprocessing {path}")
            if self.convert:
                image = tifffile.imread(path)
            else:
                with h5py.File(path, "r") as f:
                    image = f[self.dataset][...]

            image = image.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            image = self.filter(image, self.param)
            image = self.down_sample(image, self.factor, self.order)

            if self.safe:
                os.makedirs(f"{os.path.dirname(path)}/{self.safe_directory}/", exist_ok=True)
                file_name = os.path.splitext(os.path.basename(path))[0]
                h5_file_path = f"{os.path.dirname(path)}/{self.safe_directory}/{file_name}.h5"
            else:
                h5_file_path = path

            with h5py.File(h5_file_path, "w") as f:
                image = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))
                image = image.astype(np.uint8)
                f.create_dataset(self.dataset, data=image, chunks=True,  compression='gzip')

    @staticmethod
    def dummy(image, param):
        return image

    @staticmethod
    def down_sample(image, factor, order):
        if np.prod(factor) == 1:
            return image
        else:
            return zoom(image, zoom=factor, order=order)

    @staticmethod
    def median(image, radius):
        return discMedian(image, radius)

    @staticmethod
    def gaussian(image, sigma):
        return gaussianSmoothing(image, sigma)
