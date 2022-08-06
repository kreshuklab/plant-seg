from concurrent.futures import Future

from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple, LabelsData

from plantseg.napari.widget.utils import start_threading_process
from plantseg.segmentation.functional import gasp, dt_watershed


@magicgui(call_button='Run GASP', beta={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_gasp(image: ImageData, labels: LabelsData,
                beta: float = 0.5,
                minsize: int = 100) -> Future[LayerDataTuple]:
    image = image.astype('float32')
    def func():
        out = gasp(boundary_pmaps=image, superpixels=labels, beta=beta, post_minsize=minsize)
        return out, {'name': 'GASP'}, 'labels'

    return start_threading_process(func)


@magicgui(call_button='Run WS', threshold={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_dt_ws(image: ImageData, threshold: float = 0.5) -> Future[LayerDataTuple]:

    image = image.astype('float32')

    def func():
        out = dt_watershed(boundary_pmaps=image, threshold=threshold)
        return out, {'name': 'dt WS'}, 'labels'

    return start_threading_process(func)
