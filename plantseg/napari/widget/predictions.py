from concurrent.futures import Future
from typing import Tuple
from functools import partial
from magicgui import magicgui
from napari.layers import Image
from napari.types import ImageData, LayerDataTuple

from plantseg.gui import list_models
from plantseg.napari.widget.utils import start_threading_process
from plantseg.predictions.functional import unet_predictions
from plantseg.predictions.utils import STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE


@magicgui(call_button='Run Predictions',
          model_name={"choices": list_models()},
          stride={"choices": [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]},
          device={"choices": ['cpu', 'cuda']}
          )
def widget_unet_predictions(image: Image,
                            model_name,
                            patch_size: Tuple[int, int, int] = (80, 160, 160),
                            stride: str = STRIDE_ACCURATE,
                            device: str = 'cuda',) -> Future[LayerDataTuple]:

    out_name = f'{image.name}_{model_name}'
    inputs_names = (image.name, )
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
