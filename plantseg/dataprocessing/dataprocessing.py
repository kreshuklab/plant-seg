import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import ball, disk
from vigra.filters import gaussianSmoothing

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import GenericPipelineStep


def _rescale(image, factor, order):
    if np.array_equal(factor, [1, 1, 1]):
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def _median(image, radius):
    if image.shape[0] == 1:
        shape = image.shape
        median_image = median(image[0], disk(radius))
        return median_image.reshape(shape)
    else:
        return median(image, ball(radius))


def _gaussian(image, sigma):
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma)


def _no_filter(image, param):
    return image


class DataPostProcessing3D(GenericPipelineStep):
    def __init__(self,
                 input_paths,
                 input_type="labels",
                 output_type="labels",
                 save_directory="PostProcessing",
                 factor=None,
                 out_ext=".h5",
                 state=True):
        if factor is None:
            factor = [1, 1, 1]

        h5_input_key = "segmentation" if input_type == "labels" else "predictions"
        h5_output_key = h5_input_key

        super().__init__(input_paths,
                         h5_input_key=h5_input_key,
                         h5_output_key=h5_output_key,
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=save_directory,
                         out_ext=out_ext,
                         state=state)

        # rescaling
        self.factor = factor
        # spline order, use 2 for 'segmentation' and 0 for 'predictions'
        self.order = 0 if input_type == "labels" else 2

    def process(self, image):
        gui_logger.info("Postprocessing files...")

        image = _rescale(image, self.factor, self.order)

        return image


class DataPreProcessing3D(GenericPipelineStep):
    def __init__(self,
                 input_paths,
                 input_type="data_float32",
                 output_type="data_uint8",
                 save_directory="PreProcessing",
                 factor=None,
                 filter_type=None,
                 filter_param=None,
                 state=True):

        super().__init__(input_paths,
                         h5_input_key="raw",
                         h5_output_key="raw",
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=save_directory,
                         out_ext=".h5",
                         state=state)

        if factor is None:
            factor = [1, 1, 1]

        # TODO: remove below code duplication
        # rescaling
        self.factor = factor
        # spline order, use 2 for 'segmentation' and 0 for 'predictions'
        self.order = 0 if input_type == "labels" else 2

        # configure filter
        if filter_type is not None:
            assert filter_type in ["median", "gaussian"]
            assert filter_param is not None

            if filter_type == "median":
                self.filter = _median
            else:
                self.filter = _gaussian

            self.filter_param = filter_param
        else:
            self.filter = _no_filter
            self.filter_param = 0

    def process(self, image):
        gui_logger.info(f"Preprocessing files...")

        image = self.filter(image, self.filter_param)
        image = _rescale(image, self.factor, self.order)

        return image
