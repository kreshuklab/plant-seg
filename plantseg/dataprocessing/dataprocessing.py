import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import ball, disk
from skimage.transform import resize
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
                 state=True,
                 save_raw=False,
                 output_shapes=None):
        if factor is None:
            factor = [1, 1, 1]

        h5_output_key = "segmentation" if input_type == "labels" else "predictions"

        super().__init__(input_paths,
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=save_directory,
                         out_ext=out_ext,
                         state=state,
                         h5_output_key=h5_output_key,
                         save_raw=save_raw)

        # rescaling
        self.factor = factor
        # output shapes (override factor if provided)
        self.output_shapes = output_shapes
        # spline order, use 2 for 'segmentation' and 0 for 'predictions'
        self.order = 0 if input_type == "labels" else 2
        # count processed images
        self.img_count = 0

    def process(self, image):
        gui_logger.info("Postprocessing files...")

        if self.output_shapes is not None:
            # use resize
            output_shape = self.output_shapes[self.img_count]
            gui_logger.info(f"Resizing image {self.img_count} from shape: {image.shape} to shape: {output_shape}")
            image = resize(image, output_shape, self.order)
        else:
            # use standard rescaling
            image = _rescale(image, self.factor, self.order)

        self.img_count += 1
        return image


def _parse_crop(crop_str):
    crop_str = crop_str.replace('[', '').replace(']', '')
    return tuple(
        (slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip())) for
        part in crop_str.split(','))


class DataPreProcessing3D(GenericPipelineStep):
    def __init__(self,
                 input_paths,
                 input_type="data_float32",
                 output_type="data_uint8",
                 save_directory="PreProcessing",
                 factor=None,
                 filter_type=None,
                 filter_param=None,
                 state=True,
                 crop=None):

        super().__init__(input_paths,
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=save_directory,
                         out_ext=".h5",
                         state=state,
                         h5_output_key='raw')

        if factor is None:
            factor = [1, 1, 1]

        if crop is not None:
            crop = _parse_crop(crop)
        self.crop = crop

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
        if self.crop is not None:
            gui_logger.info(f"Cropping input image to: {self.crop}")
            image = image[self.crop]

        image = self.filter(image, self.filter_param)
        image = _rescale(image, self.factor, self.order)

        return image
