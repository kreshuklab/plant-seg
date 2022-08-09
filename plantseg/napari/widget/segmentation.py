from concurrent.futures import Future
from functools import partial
from magicgui import magicgui
from napari.types import ImageData, LayerDataTuple, LabelsData
from napari.layers import Labels, Image, Layer
from typing import Union, Tuple, Callable
from plantseg.napari.widget.utils import start_threading_process
from plantseg.segmentation.functional import gasp, multicut, dt_watershed
from plantseg.segmentation.functional import lifted_multicut_from_nuclei_segmentation, lifted_multicut_from_nuclei_pmaps


def _generic_clustering(image: Image, labels: Labels,
                        beta: float = 0.5,
                        minsize: int = 100,
                        name: str = "GASP",
                        agg_func: Callable = gasp) -> Future[LayerDataTuple]:
    out_name = f'{image.name}_{name}'
    inputs_names = (image.name, labels.name)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'labels'

    func = partial(agg_func, beta=beta, post_minsize=minsize)

    return start_threading_process(func,
                                   func_kwargs={'boundary_pmaps': image.data,
                                                'superpixels': labels.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )


@magicgui(call_button='Run GASP', beta={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_gasp(image: Image, labels: Labels,
                beta: float = 0.5,
                minsize: int = 100) -> Future[LayerDataTuple]:
    return _generic_clustering(image, labels, beta=beta, minsize=minsize, name='GASP', agg_func=gasp)


@magicgui(call_button='Run MultiCut', beta={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_multicut(image: Image, labels: Labels,
                    beta: float = 0.5,
                    minsize: int = 100) -> Future[LayerDataTuple]:
    return _generic_clustering(image, labels, beta=beta, minsize=minsize, name='MultiCut', agg_func=multicut)


@magicgui(call_button='Run Lifted MultiCut', beta={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
def widget_lifted_multicut(image: Image,
                           nuclei: Layer,
                           labels: Labels,
                           beta: float = 0.5,
                           minsize: int = 100) -> Future[LayerDataTuple]:
    if isinstance(nuclei, Image):
        lmc = lifted_multicut_from_nuclei_pmaps
        extra_key = 'nuclei_pmaps'
    elif isinstance(nuclei, Labels):
        lmc = lifted_multicut_from_nuclei_segmentation
        extra_key = 'nuclei_seg'
    else:
        raise ValueError(f'{nuclei} must be either an image or a labels layer')

    out_name = f'{image.name}_Lifted_MultiCut'
    inputs_names = (image.name, nuclei.name, labels.name)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'labels'

    func = partial(lmc, superpixels=labels, beta=beta, post_minsize=minsize)

    return start_threading_process(func,
                                   func_kwargs={'boundary_pmaps': image.data,
                                                extra_key: nuclei.data,
                                                'superpixels': labels.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )


@magicgui(call_button='Run WS',
          stacked={'widget_type': 'RadioButtons', 'orientation': 'horizontal', 'choices': ['2D', '3D']},
          threshold={"widget_type": "FloatSlider", "max": 1., 'min': 0.}, )
def widget_dt_ws(image: Image,
                 stacked: str = '2D',
                 threshold: float = 0.5,
                 min_size: int = 100,
                 sigma_seeds: float = .2,
                 sigma_weights: float = 2.,
                 alpha: float = 1.,
                 use_pixel_pitch: bool = False,
                 pixel_pitch: Tuple[int, int, int] = (1, 1, 1),
                 apply_nonmax_suppression: bool = False) -> Future[LayerDataTuple]:
    out_name = f'{image.name}_dt_WS'
    inputs_names = (image.name,)
    layer_kwargs = {'name': out_name, 'scale': image.scale}
    layer_type = 'labels'

    stacked = False if stacked == '3D' else True
    pixel_pitch = pixel_pitch if use_pixel_pitch else None
    func = partial(dt_watershed,
                   threshold=threshold,
                   min_size=min_size,
                   stacked=stacked,
                   sigma_seeds=sigma_seeds,
                   sigma_weights=sigma_weights,
                   alpha=alpha,
                   pixel_pitch=pixel_pitch,
                   apply_nonmax_suppression=apply_nonmax_suppression
                   )
    return start_threading_process(func,
                                   func_kwargs={'boundary_pmaps': image.data},
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   )
