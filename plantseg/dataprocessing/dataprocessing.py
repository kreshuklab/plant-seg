from skimage.transform import resize

from plantseg.dataprocessing.functional.dataprocessing import image_rescale, image_median, image_gaussian_smoothing, \
    image_crop
from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import GenericPipelineStep


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
            image = image_rescale(image, self.factor, self.order)

        self.img_count += 1
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
                self.filter = image_median
            else:
                self.filter = image_gaussian_smoothing

            self.filter_param = filter_param
        else:
            self.filter = _no_filter
            self.filter_param = 0

    def process(self, image):
        gui_logger.info(f"Preprocessing files...")
        if self.crop is not None:
            gui_logger.info(f"Cropping input image to: {self.crop}")
            image = image_crop(image, self.crop)

        image = self.filter(image, self.filter_param)
        image = image_rescale(image, self.factor, self.order)

        return image
