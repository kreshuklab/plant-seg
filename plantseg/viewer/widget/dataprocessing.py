import math
from concurrent.futures import Future
from functools import partial
from typing import Tuple, Union

import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes, Layer
from napari.types import LayerDataTuple

from plantseg.dataprocessing.functional import image_gaussian_smoothing, image_rescale
from plantseg.dataprocessing.functional.dataprocessing import compute_scaling_factor, compute_scaling_voxelsize
from plantseg.dataprocessing.functional.labelprocessing import relabel_segmentation as _relabel_segmentation
from plantseg.dataprocessing.functional.labelprocessing import set_background_to_value
from plantseg.utils import list_models, get_model_resolution
from plantseg.viewer.widget.utils import start_threading_process, build_nice_name, layer_properties


@magicgui(call_button='Run Gaussian Smoothing',
          image={'label': 'Image',
                 'tooltip': 'Image layer to apply the smoothing.'},
          sigma={'label': 'Sigma',
                 'widget_type': 'FloatSlider',
                 'tooltip': 'Define the size of the gaussian smoothing kernel. '
                            'The larger the more blurred will be the output image.',
                 'max': 5.,
                 'min': 0.})
def widget_gaussian_smoothing(image: Image,
                              sigma: float = 1.,
                              ) -> Future[LayerDataTuple]:
    out_name = build_nice_name(image.name, 'GaussianSmoothing')
    inputs_kwarg = {'image': image.data}
    step_kwargs = {'sigma': sigma}
    inputs_names = (image.name,)
    layer_kwargs = layer_properties(name=out_name, scale=image.scale, metadata=image.metadata)
    layer_type = 'image'
    func = partial(image_gaussian_smoothing, **step_kwargs)

    return start_threading_process(func,
                                   runtime_kwargs=inputs_kwarg,
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='Gaussian Smoothing',
                                   )


@magicgui(call_button='Run Image Rescaling',
          image={'label': 'Image or Label',
                 'tooltip': 'Layer to apply the rescaling.'},
          type_of_refactor={'label': 'Type of refactor',
                            'tooltip': 'Select the mode of finding the right rescaling factor.',
                            'widget_type': 'RadioButtons',
                            'orientation': 'vertical',
                            'choices': ['Rescaling factor',
                                        'Voxel size',
                                        'Same as Reference Layer',
                                        'Same as Reference Model']},
          rescaling_factor={'label': 'Rescaling factor',
                            'tooltip': 'Define the scaling factor to use for resizing the input image.'},
          out_voxel_size={'label': 'Out voxel size',
                          'tooltip': 'Define the output voxel size. Units are same as imported, '
                                     '(if units are missing default is "um").'},
          reference_layer={'label': 'Reference layer',
                           'tooltip': 'Rescale to same voxel size as selected layer.'},
          reference_model={'label': 'Reference model',
                           'tooltip': 'Rescale to same voxel size as selected model.',
                           'choices': list_models()},
          order={'label': 'Interpolation order',
                 'tooltip': '0 for nearest neighbours (default for labels), 1 for linear, 2 for bilinear.',
                 })
def widget_rescaling(image: Layer,
                     type_of_refactor: str = 'Rescaling factor',
                     rescaling_factor: Tuple[float, float, float] = (1., 1., 1.),
                     out_voxel_size: Tuple[float, float, float] = (1., 1., 1.),
                     reference_layer: Union[None, Layer] = None,
                     reference_model: str = list_models()[0],
                     order: int = 1,
                     ) -> Future[LayerDataTuple]:
    if isinstance(image, Image):
        pass

    elif isinstance(image, Labels):
        order = 0

    else:
        raise ValueError(f'{type(image)} cannot be rescaled, please use Image layers or Labels layers')

    current_resolution = image.scale
    if type_of_refactor == 'Voxel size (um)':
        rescaling_factor = compute_scaling_factor(current_resolution, out_voxel_size)

    elif type_of_refactor == 'Same as Reference Layer':
        out_voxel_size = reference_layer.scale
        rescaling_factor = compute_scaling_factor(current_resolution, reference_layer.scale)

    elif type_of_refactor == 'Same as Reference Model':
        out_voxel_size = get_model_resolution(reference_model)
        rescaling_factor = compute_scaling_factor(current_resolution, out_voxel_size)

    else:
        out_voxel_size = compute_scaling_voxelsize(current_resolution, scaling_factor=rescaling_factor)

    out_name = build_nice_name(image.name, 'Rescaled')
    inputs_kwarg = {'image': image.data}
    inputs_names = (image.name,)
    step_kwargs = {'factor': rescaling_factor, 'order': order}
    layer_kwargs = layer_properties(name=out_name,
                                    scale=out_voxel_size,
                                    metadata={**image.metadata,
                                              **{'original_voxel_size': current_resolution}})
    layer_type = 'image'

    return start_threading_process(image_rescale,
                                   runtime_kwargs=inputs_kwarg,
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   step_name='Rescaling',
                                   layer_type=layer_type,
                                   )


def _compute_slices(rectangle, crop_z, shape):
    z_start = max(rectangle[0, 0] - crop_z // 2, 0)
    z_end = min(rectangle[0, 0] + math.ceil(crop_z / 2), shape[0])
    z_slice = slice(z_start, z_end)

    x_start = max(rectangle[0, 1], 0)
    x_end = min(rectangle[2, 1], shape[1])
    x_slice = slice(x_start, x_end)

    y_start = max(rectangle[0, 2], 0)
    y_end = min(rectangle[2, 2], shape[2])
    y_slice = slice(y_start, y_end)

    return z_slice, x_slice, y_slice


def _cropping(data, crop_slices):
    return data[crop_slices]


@magicgui(call_button='Run Cropping',
          image={'label': 'Image or Label',
                 'tooltip': 'Layer to apply the rescaling.'},
          crop_roi={'label': 'Crop ROI',
                    'tooltip': 'This must be a shape layer with a rectangle XY overlaying the area to crop.'},
          crop_z={'label': 'Z slices',
                  'tooltip': 'Numer of z slices to take next to the current selection.'},
          )
def widget_cropping(image: Layer,
                    crop_roi: Union[Shapes, None] = None,
                    crop_z: int = 1,
                    ) -> Future[LayerDataTuple]:
    assert len(crop_roi.shape_type) == 1, "Only one rectangle should be used for cropping"
    assert crop_roi.shape_type[0] == 'rectangle', "Only a rectangle shape should be used for cropping"

    out_name = build_nice_name(image.name, 'cropped')
    inputs_names = (image.name,)
    layer_kwargs = layer_properties(name=out_name,
                                    scale=image.scale,
                                    metadata=image.metadata)
    layer_type = 'image'

    rectangle = crop_roi.data[0].astype('int64')

    crop_slices = _compute_slices(rectangle, crop_z, image.data.shape)

    return start_threading_process(_cropping,
                                   runtime_kwargs={'data': image.data},
                                   statics_kwargs={'crop_slices': crop_slices},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='Cropping',
                                   skip_dag=True,
                                   )


def _two_layers_operation(data1, data2, operation, weights: float = 0.5):
    if operation == 'Mean':
        return weights * data1 + (1. - weights) * data2
    elif operation == 'Maximum':
        return np.maximum(data1, data2)
    else:
        return np.minimum(data1, data2)


@magicgui(call_button='Run Merge Layers',
          image1={'label': 'Image 1'},
          image2={'label': 'Image 2'},
          operation={'label': 'Operation',
                     'tooltip': 'Operation used to merge the two layers.',
                     'widget_type': 'RadioButtons',
                     'orientation': 'horizontal',
                     'choices': ['Mean',
                                 'Maximum',
                                 'Minimum']},
          weights={'label': 'Mean weights',
                   'widget_type': 'FloatSlider', 'max': 1., 'min': 0.},
          )
def widget_add_layers(image1: Image,
                      image2: Image,
                      operation: str = 'Maximum',
                      weights: float = 0.5,
                      ) -> Future[LayerDataTuple]:
    out_name = build_nice_name(f'{image1.name}-{image2.name}', operation)
    inputs_names = (image1.name, image2.name)
    layer_kwargs = layer_properties(name=out_name,
                                    scale=image1.scale,
                                    metadata=image1.metadata)
    layer_type = 'image'
    step_kwargs = dict(weights=weights, operation=operation)
    assert image1.data.shape == image2.data.shape

    return start_threading_process(_two_layers_operation,
                                   runtime_kwargs={'data1': image1.data, 'data2': image2.data},
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='Merge Layers',
                                   )


def _label_processing(segmentation, set_bg_to_0, relabel_segmentation):
    if relabel_segmentation:
        segmentation = _relabel_segmentation(segmentation)

    if set_bg_to_0:
        segmentation = set_background_to_value(segmentation, value=0)

    return segmentation


@magicgui(call_button='Run Label processing',
          segmentation={'label': 'Segmentation',
                        'tooltip': 'Segmentation can be any label layer.'},
          set_bg_to_0={'label': 'Set background to 0',
                       'tooltip': 'Set the largest idx in the image to zero.'},
          relabel_segmentation={'label': 'Relabel Segmentation',
                                'tooltip': 'Relabel segmentation contiguously to avoid labels clash.'}
          )
def widget_label_processing(segmentation: Labels,
                            set_bg_to_0: bool = True,
                            relabel_segmentation: bool = True,
                            ) -> Future[LayerDataTuple]:
    if relabel_segmentation and 'bboxes' in segmentation.metadata.keys():
        del segmentation.metadata['bboxes']

    out_name = build_nice_name(segmentation.name, 'Processed')
    inputs_kwarg = {'segmentation': segmentation.data}
    inputs_names = (segmentation.name,)
    layer_kwargs = layer_properties(name=out_name,
                                    scale=segmentation.scale,
                                    metadata=segmentation.metadata)
    layer_type = 'labels'
    step_kwargs = dict(set_bg_to_0=set_bg_to_0, relabel_segmentation=relabel_segmentation)

    return start_threading_process(_label_processing,
                                   runtime_kwargs=inputs_kwarg,
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='Label Processing',
                                   )
