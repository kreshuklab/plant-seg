from concurrent.futures import Future

from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple
from napari.layers import Image, Labels, Shapes
from typing import Tuple, Union

from plantseg.dataprocessing.functional import image_gaussian_smoothing, image_crop, image_rescale
from plantseg.napari.widget.utils import start_threading_process, choices_of_image_layers


@magicgui(call_button='Run Gaussian Smoothing',
          sigma={"widget_type": "FloatSlider", "max": 5., 'min': 0.1})
def widget_image_gaussian_smoothing(image: Image,
                                    sigma: float = 0.5) -> Future[LayerDataTuple]:
    def func():
        _image = image.data
        out = image_gaussian_smoothing(image=_image, sigma=sigma)
        return out, {'name': 'Cell Boundary Raw'}, 'image'

    return start_threading_process(func)


@magicgui(call_button='Run Data Preprocessing',
          sigma={"widget_type": "FloatSlider", "max": 5., 'min': 0.1})
def widget_generic_preprocessing(image: Image,
                                 gaussian_smoothing: bool = False,
                                 sigma: float = 1.,
                                 rescale: bool = False,
                                 rescaling_factors: Tuple[float, float, float] = (1., 1., 1.),
                                 ) -> Future[LayerDataTuple]:
    def func():
        _image = image.data
        if gaussian_smoothing:
            _image = image_gaussian_smoothing(image=_image, sigma=sigma)
        if rescale:
            _image = image_rescale(image=_image, factor=rescaling_factors, order=1)
        return _image, {'name': 'Cell Boundary Raw'}, 'image'

    return start_threading_process(func)


@magicgui(call_button='Run Cropping', )
def widget_cropping(data: Labels,
                    crop_roi: Union[Shapes, None] = None,
                    crop_z: int = 1,
                    ) -> LayerDataTuple:

    assert len(crop_roi.shape_type) == 1, "Only one rectangle should be used for cropping"
    assert crop_roi.shape_type[0] == 'rectangle', "Only a rectangle shape should be used for cropping"
    rectangle = crop_roi.data[0].astype('int64')
    crop_slices = [slice(rectangle[0, 0], rectangle[2, 0]), slice(rectangle[0, 1], rectangle[2, 1])]
    data_image = data.data
    if data_image.ndim == 2:
        crop_slices = tuple(crop_slices)
    elif data_image.ndim == 3:
        crop_slices = tuple([1] + crop_slices)

    out = data.data[crop_slices]
    return out, {'name': f'cropped_{data.name}'}, 'image'


@magicgui(call_button='Run Add Layers', weights={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_add_layers(image1: Image,
                      image2: Image,
                      weights: float = 0.5,
                      ) -> Future[LayerDataTuple]:

    def func():
        out = weights * image1.data + (1. - weights) * image2.data
        return out, {'name': 'Cell Boundary Raw'}, 'image'

    return start_threading_process(func)
