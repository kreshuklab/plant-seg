from concurrent.futures import Future
from typing import Tuple

from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple

from plantseg.gui import list_models
from plantseg.napari.widget.utils import start_threading_process
from plantseg.predictions.functional import unet_predictions


@magicgui(call_button='Run Predictions', model_name={"choices": list_models()}, )
def widget_unet_predictions(image: ImageData,
                            model_name,
                            patch_size: Tuple[int, int, int] = (80, 160, 160)) -> Future[LayerDataTuple]:
    image = image.astype('float32')

    def func():
        out = unet_predictions(raw=image, model_name=model_name, patch=patch_size, device='cpu')
        return out, {'name': model_name}, 'image'

    return start_threading_process(func)
