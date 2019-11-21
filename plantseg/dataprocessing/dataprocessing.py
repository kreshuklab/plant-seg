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
    def __init__(self, config, paths, dataset="raw"):
        self.paths = paths

        # convert from tiff
        self.safe_directory = "postprocessing"
        self.convert = config["tiff"]
        self.dataset = dataset

        # rescaling
        self.factor = config["factor"]
        self.order = config["order"]

    def __call__(self, ):
        for path in self.paths:
            print(f"Postprocessing {path}")
            # Load h5 from predictions or segmentation
            with h5py.File(path, "r") as f:
                image = f[self.dataset][...]

            # Resample
            image = image if image.ndim == 3 else image[0]
            image = self.down_sample(image, self.factor, self.order)

            # Save as h5 of as tiff
            os.makedirs(f"{ os.path.dirname(path)}/{ self.safe_directory}/", exist_ok=True)
            file_name = os.path.split(os.path.basename(path))
            h5_file_path = f"{os.path.dirname(path)}/{self.safe_directory}/{file_name}"
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
    def __init__(self, config, paths, dataset="raw"):
        self.paths = paths

        # convert from tiff
        self.safe_directory = config["save_directory"]
        self.convert = config["extension"]
        self.dataset = dataset

        if "filter" in config.keys():
            # filters
            if "median" == config["filter"]:
                self.param = config["filter_param"]
                self.filter = self.median

            elif "gaussian" == config["filter"]:
                self.param = config["filter_param"]
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

            # Load file
            _, ext = os.path.splitext(path)
            if ext == ".tiff" or ext == ".tif":
                image = tifffile.imread(path)
            elif ext == ".hdf" or ext == ".h5" or ext == ".hd5":
                with h5py.File(path, "r") as f:
                    image = f[self.dataset][...]
            else:
                print("Data extension not understood")
                raise NotImplementedError

            # Normalize
            image = image.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            # Apply filters
            image = self.filter(image, self.param)
            image = self.down_sample(image, self.factor, self.order)

            # Save file
            os.makedirs(f"{os.path.dirname(path)}/{self.safe_directory}/", exist_ok=True)
            file_name = os.path.splitext(os.path.basename(path))[0]
            h5_file_path = f"{os.path.dirname(path)}/{self.safe_directory}/{file_name}.h5"

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
