from concurrent.futures import Future
from functools import partial
from typing import Tuple, Union

import math
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes, Layer
from napari.types import LayerDataTuple

from plantseg.dataprocessing.functional import image_gaussian_smoothing, image_rescale
from plantseg.napari.widget.utils import start_threading_process


def _generic_preprocessing(image_data, sigma, gaussian_smoothing, rescale, rescaling_factors):
    if gaussian_smoothing:
        image_data = image_gaussian_smoothing(image=image_data, sigma=sigma)
    if rescale:
        image_data = image_rescale(image=image_data, factor=rescaling_factors, order=1)

    return image_data


@magicgui(call_button='Run Data Preprocessing',
          sigma={"widget_type": "FloatSlider", "max": 5., 'min': 0.1})
def widget_generic_preprocessing(image: Image,
                                 gaussian_smoothing: bool = False,
                                 sigma: float = 1.,
                                 rescale: bool = False,
                                 rescaling_factors: Tuple[float, float, float] = (1., 1., 1.),
                                 ) -> Future[LayerDataTuple]:

    out_name = f'{image.name}_processed'
    inputs_kwarg = {'image_data': image.data}
    inputs_names = (image.name, )
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'image'
    func = partial(_generic_preprocessing, sigma=sigma,
                   gaussian_smoothing=gaussian_smoothing,
                   rescale=rescale,
                   rescaling_factors=rescaling_factors)
    return start_threading_process(func,
                                   func_kwargs=inputs_kwarg,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )


def _cropping(data, crop_slices):
    return data[crop_slices]


@magicgui(call_button='Run Cropping', )
def widget_cropping(image: Layer,
                    crop_roi: Union[Shapes, None] = None,
                    crop_z: int = 1,
                    ) -> Future[LayerDataTuple]:

    assert len(crop_roi.shape_type) == 1, "Only one rectangle should be used for cropping"
    assert crop_roi.shape_type[0] == 'rectangle', "Only a rectangle shape should be used for cropping"

    out_name = f'{image.name}_cropped'
    inputs_names = (image.name,)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'image'

    rectangle = crop_roi.data[0].astype('int64')
    crop_slices = [slice(rectangle[0, 0] - crop_z//2, rectangle[0, 0] + math.ceil(crop_z/2)),
                   slice(rectangle[0, 1], rectangle[2, 1]),
                   slice(rectangle[0, 2], rectangle[2, 2])]

    func = partial(_cropping, crop_slices=crop_slices)
    return start_threading_process(func,
                                   func_kwargs={'data': image.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   skip_dag=True,
                                   )


def _add_two_layers(data1, data2, weights: float = 0.5):
    return weights * data1 + (1. - weights) * data2


@magicgui(call_button='Run Add Layers', weights={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_add_layers(image1: Image,
                      image2: Image,
                      weights: float = 0.5,
                      ) -> Future[LayerDataTuple]:

    out_name = f'{image1.name}_{image2.name}'
    inputs_names = (image1.name, image2.name)
    layer_kwargs = {'name': out_name, 'scale': image1.scale}
    layer_type = 'image'

    func = partial(_add_two_layers, weights=weights)
    return start_threading_process(func,
                                   func_kwargs={'data1': image1.data, 'data2': image2.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )
