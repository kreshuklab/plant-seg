from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Tuple, List

import torch.cuda
from magicgui import magicgui
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.types import LayerDataTuple

from plantseg.dataprocessing.functional import image_gaussian_smoothing
from plantseg.predictions.functional import unet_predictions
from plantseg.predictions.functional.utils import STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE
from plantseg.utils import list_models, add_custom_model
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.utils import start_threading_process, create_layer_name, layer_properties

ALL_CUDA_DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
MPS = ['mps'] if torch.backends.mps.is_available() else []
ALL_DEVICES = ALL_CUDA_DEVICES + MPS + ['cpu']


def unet_predictions_wrapper(raw, device, **kwargs):
    """
    Wrapper to run unet_predictions in a thread_worker, this is needed to allow the user to select the device
    in the headless mode.
    """
    return unet_predictions(raw, device=device, **kwargs)


@magicgui(call_button='Run Predictions',
          image={'label': 'Image',
                 'tooltip': 'Raw image to be processed with a neural network.'},
          model_name={'label': 'Select model',
                      'tooltip': 'Select a pretrained model.',
                      'choices': list_models()},
          patch_size={'label': 'Patch size',
                      'tooltip': 'Patch size use to processed the data.'},
          stride={'label': 'Stride',
                  'choices': [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_unet_predictions(image: Image,
                            model_name: str,
                            patch_size: Tuple[int, int, int] = (80, 160, 160),
                            stride: str = STRIDE_ACCURATE,
                            device: str = ALL_DEVICES[0], ) -> Future[LayerDataTuple]:
    out_name = create_layer_name(image.name, model_name)

    inputs_names = (image.name, 'device')

    layer_kwargs = layer_properties(name=out_name,
                                    scale=image.scale,
                                    metadata=image.metadata)

    layer_kwargs['metadata']['pmap'] = True  # this is used to warn the user that the layer is a pmap

    layer_type = 'image'
    step_kwargs = dict(model_name=model_name, stride=stride, patch=patch_size)

    return start_threading_process(unet_predictions_wrapper,
                                   runtime_kwargs={'raw': image.data, 'device': device},
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='UNet Predictions',
                                   )


def _compute_multiple_predictions(image, patch_size, stride, device):
    out_layers = []
    for i, model_name in enumerate(list_models()):

        napari_formatted_logging(f'Running UNet Predictions: {model_name} {i}/{len(list_models())}',
                                 thread='UNet Grid Predictions')

        out_name = create_layer_name(image.name, model_name)
        layer_kwargs = layer_properties(name=out_name,
                                        scale=image.scale,
                                        metadata=image.metadata)
        layer_kwargs['metadata']['pmap'] = True  # this is used to warn the user that the layer is a pmap
        layer_type = 'image'
        try:
            pmap = unet_predictions(raw=image.data,
                                    model_name=model_name,
                                    stride=stride,
                                    patch=patch_size,
                                    device=device)
            out_layers.append((pmap, layer_kwargs, layer_type))

        except Exception as e:
            print(f'Error while processing: {model_name}')

    return out_layers


@magicgui(call_button='Try all Available Models',
          image={'label': 'Image',
                 'tooltip': 'Raw image to be processed with a neural network.'},
          patch_size={'label': 'Patch size',
                      'tooltip': 'Patch size use to processed the data.'},
          stride={'label': 'Stride',
                  'choices': [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_test_all_unet_predictions(image: Image,
                                     patch_size: Tuple[int, int, int] = (80, 160, 160),
                                     stride: str = STRIDE_ACCURATE,
                                     device: str = ALL_DEVICES[0]) -> Future[List[LayerDataTuple]]:
    func = thread_worker(partial(_compute_multiple_predictions,
                                 image=image,
                                 patch_size=patch_size,
                                 stride=stride,
                                 device=device))

    future = Future()

    def on_done(result):
        future.set_result(result)

    worker = func()
    worker.returned.connect(on_done)
    worker.start()
    return future


def _compute_iterative_predictions(pmap, model_name, num_iterations, sigma, patch_size, stride, device):
    func = partial(unet_predictions, model_name=model_name, stride=stride, patch=patch_size, device=device)
    for i in range(num_iterations - 1):
        pmap = func(pmap)
        pmap = image_gaussian_smoothing(image=pmap, sigma=sigma)

    pmap = func(pmap)
    return pmap


@magicgui(call_button='Run Iterative Predictions',
          image={'label': 'Image',
                 'tooltip': 'Raw image to be processed with a neural network.'},
          model_name={'label': 'Select model',
                      'tooltip': 'Select a pretrained model.',
                      'choices': list_models()},
          num_iterations={'label': 'Num. of iterations',
                          'tooltip': 'Nuber of iterations the model will run.'},
          sigma={'label': 'Sigma',
                 'widget_type': 'FloatSlider',
                 'tooltip': 'Define the size of the gaussian smoothing kernel. '
                            'The larger the more blurred will be the output image.',
                 'max': 5.,
                 'min': 0.},
          patch_size={'label': 'Patch size',
                      'tooltip': 'Patch size use to processed the data.'},
          stride={'label': 'Stride',
                  'choices': [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_iterative_unet_predictions(image: Image,
                                      model_name: str,
                                      num_iterations: int = 2,
                                      sigma: float = 1.0,
                                      patch_size: Tuple[int, int, int] = (80, 160, 160),
                                      stride: str = STRIDE_ACCURATE,
                                      device: str = ALL_DEVICES[0]) -> Future[LayerDataTuple]:
    out_name = create_layer_name(image.name, f'iterative-{model_name}-x{num_iterations}')
    inputs_names = (image.name,)
    layer_kwargs = layer_properties(name=out_name,
                                    scale=image.scale,
                                    metadata=image.metadata)
    layer_kwargs['metadata']['pmap'] = True  # this is used to warn the user that the layer is a pmap
    layer_type = 'image'
    step_kwargs = dict(model_name=model_name,
                       num_iterations=num_iterations,
                       sigma=sigma,
                       patch_size=patch_size,
                       stride=stride,
                       device=device)

    return start_threading_process(_compute_iterative_predictions,
                                   runtime_kwargs={'pmap': image.data},
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='UNet Iterative Predictions',
                                   )


@magicgui(call_button='Add Custom Model',
          new_model_name={'label': 'New model name'},
          model_location={'label': 'Model location',
                          'mode': 'd'},
          resolution={'label': 'Resolution'},
          description={'label': 'Description'},
          )
def widget_add_custom_model(new_model_name: str = '',
                            model_location: Path = Path.home(),
                            resolution: Tuple[float, float, float] = (1., 1., 1.),
                            description: str = '') -> None:
    new_model_name = 'custom_model' if new_model_name == '' else new_model_name
    add_custom_model(new_model_name=new_model_name,
                     location=model_location,
                     resolution=resolution,
                     description=description)
