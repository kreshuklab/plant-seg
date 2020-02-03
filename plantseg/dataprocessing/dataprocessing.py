import time
import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import ball
from vigra.filters import gaussianSmoothing

from plantseg import GenericProcessing


def _dummy(image, param):
    return image


def _rescale(image, factor, order):
    order = int(order)
    if np.prod(factor) == 1:
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def _median(image, radius):
    return median(image, ball(radius))


def _gaussian(image, sigma):
    return gaussianSmoothing(image, sigma)


class DataPostProcessing3D(GenericProcessing):
    def __init__(self,
                 paths,
                 config,
                 data_type="labels"):

        save_directory = config["save_directory"] if "save_directory" in config else "PostProcessing"
        output_type = config["output_type"] if "output_type" in config else data_type

        super().__init__(paths,
                         input_type=data_type,
                         output_type=output_type,
                         save_directory=save_directory)
        self.paths = paths

        # convert from tiff
        self.out_ext = ".tiff" if config["tiff"] else ".h5"

        # rescaling
        self.factor = config["factor"]
        self.order = config["order"]

        self.data_type = data_type
        self.dataset = "segmentation" if data_type == "labels" else "predictions"

    def __call__(self, ):
        for path in self.paths:
            runtime = time.time()
            print(f"Postprocessing {path}")
            # Load h5 from predictions or segmentation
            output_path, exist = self.create_output_path(path,
                                                         prefix="",
                                                         out_ext=self.out_ext)

            image = self.load_stack(path)
            image = _rescale(image, self.factor, self.order)
            self.save_output(image, output_path, dataset=self.dataset)
            self.outputs_paths.append(output_path)

            runtime = time.time() - runtime
            print(f" - PostProcessing took {runtime:.2f} s")

        # returns input paths for postprocessing are not updated.
        return self.paths


class DataPreProcessing3D(GenericProcessing):
    def __init__(self,
                 paths,
                 config,
                 data_type="data_float32"):

        save_directory = config["save_directory"] if "save_directory" in config else "PreProcessing"
        output_type = config["output_type"] if "output_type" in config else "data_uint8"

        super().__init__(paths,
                         input_type=data_type,
                         output_type=output_type,
                         save_directory=save_directory)
        self.paths = paths

        # convert from tiff
        self.out_ext = ".h5"

        # rescaling
        self.factor = config["factor"]
        self.order = config["order"]

        # filter
        if config["filter"]["state"]:
            # filters
            if "median" == config["filter"]["type"]:
                self.param = config["filter"]["param"]
                self.filter = _median

            elif "gaussian" == config["filter"]["type"]:
                self.param = config["filter"]["param"]
                self.filter = _gaussian
            else:
                raise NotImplementedError
        else:
            self.param = 0
            self.filter = _dummy

        self.data_type = data_type
        self.dataset = "raw"

    def __call__(self, ):
        for path in self.paths:
            runtime = time.time()
            print(f"PreProcessing {path}")
            # Load h5 from predictions or segmentation
            output_path, exist = self.create_output_path(path,
                                                         prefix="",
                                                         out_ext=self.out_ext)

            image = self.load_stack(path)
            image = self.filter(image, self.param)
            image = _rescale(image, self.factor, self.order)

            self.save_output(image, output_path, dataset=self.dataset)
            self.outputs_paths.append(output_path)

            runtime = time.time() - runtime
            print(f" - PreProcessing took {runtime:.2f} s")

        return self.outputs_paths
