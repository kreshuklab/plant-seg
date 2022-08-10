from concurrent.futures import Future
from typing import Tuple, List
from functools import partial
from magicgui import magicgui
from napari.layers import Image, Points
from napari.types import ImageData, LayerDataTuple
from napari.qt.threading import thread_worker
from plantseg.gui import list_models
from plantseg.napari.widget.utils import start_threading_process, build_nice_name
from plantseg.predictions.functional import unet_predictions
from plantseg.dataprocessing.functional import image_gaussian_smoothing
from plantseg.predictions.utils import STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE


@magicgui(call_button='Run Predictions',
          model_name={"choices": list_models()},
          stride={"choices": [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={"choices": ['cpu', 'cuda']}
          )
def widget_unet_predictions(image: Image,
                            model_name: str,
                            patch_size: Tuple[int, int, int] = (80, 160, 160),
                            stride: str = STRIDE_ACCURATE,
                            device: str = 'cuda', ) -> Future[LayerDataTuple]:
    out_name = build_nice_name(image.name, model_name)

    inputs_names = (image.name,)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'image'

    func = partial(unet_predictions, model_name=model_name, stride=stride, patch=patch_size, device=device)

    return start_threading_process(func,
                                   func_kwargs={'raw': image.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )


def _compute_multiple_predictions(image, patch_size, stride, device):
    out_layers = []
    for model_name in list_models():
        out_name = build_nice_name(image.name, model_name)
        layer_kwargs = {'name': out_name, 'scale': image.scale}
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


@magicgui(call_button='Run Test Predictions',
          stride={"choices": [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={"choices": ['cpu', 'cuda']}
          )
def widget_test_all_unet_predictions(image: Image,
                                     patch_size: Tuple[int, int, int] = (80, 160, 160),
                                     stride: str = STRIDE_ACCURATE,
                                     device: str = 'cuda', ) -> Future[List[LayerDataTuple]]:
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
          model_name={"choices": list_models()},
          stride={"choices": [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={"choices": ['cpu', 'cuda']}
          )
def widget_iterative_unet_predictions(image: Image,
                                      model_name: str,
                                      num_iterations: int = 2,
                                      sigma: float = 1.,
                                      patch_size: Tuple[int, int, int] = (80, 160, 160),
                                      stride: str = STRIDE_ACCURATE,
                                      device: str = 'cuda', ) -> Future[LayerDataTuple]:

    out_name = build_nice_name(image.name, f'iterative-{model_name}-x{num_iterations}')
    inputs_names = (image.name,)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'image'

    func = partial(_compute_iterative_predictions,
                   model_name=model_name,
                   num_iterations=num_iterations,
                   sigma=sigma,
                   patch_size=patch_size,
                   stride=stride,
                   device=device)

    return start_threading_process(func,
                                   func_kwargs={'pmap': image.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )
