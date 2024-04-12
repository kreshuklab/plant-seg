from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Tuple, List

import torch.cuda
from magicgui import magicgui
from napari import Viewer
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.types import LayerDataTuple

from plantseg.dataprocessing.functional import image_gaussian_smoothing
from plantseg.predictions.functional import unet_predictions
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from plantseg.viewer.widget.segmentation import widget_agglomeration, widget_lifted_multicut, widget_simple_dt_ws
from plantseg.viewer.widget.utils import return_value_if_widget
from plantseg.viewer.widget.utils import start_threading_process, start_prediction_process, create_layer_name, layer_properties
from plantseg.viewer.widget.validation import _on_prediction_input_image_change
from plantseg.models.zoo import model_zoo

ALL = 'All'
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
          dimensionality={'label': 'Dimensionality',
                          'tooltip': 'Dimensionality of the model (2D or 3D). '
                                     'Any 2D model can be used for 3D data. If unsure, select "All".',
                          'widget_type': 'ComboBox',
                          'choices': [ALL] + model_zoo.get_unique_dimensionalities()},
          modality={'label': 'Microscopy Modality',
                    'tooltip': 'Modality of the model (e.g. confocal, light-sheet ...). If unsure, select "All".',
                    'widget_type': 'ComboBox',
                    'choices': [ALL] + model_zoo.get_unique_modalities()},
          output_type={'label': 'Prediction type',
                       'widget_type': 'ComboBox',
                       'tooltip': 'Type of prediction (e.g. cell boundaries predictions or nuclei...).'
                                  ' If unsure, select "All".',
                       'choices': [ALL] + model_zoo.get_unique_output_types()},
          model_name={'label': 'Select model',
                      'tooltip': f'Select a pretrained model. '
                                 f'Current model description: {model_zoo.get_model_description(model_zoo.list_models()[0])}',
                      'choices': model_zoo.list_models()},
          patch_size={'label': 'Patch size',
                      'tooltip': 'Patch size use to processed the data.'},
          patch_halo={'label': 'Patch halo',
                      'tooltip': 'Patch halo is extra padding for correct prediction on image boarder.'},
          single_patch={'label': 'Single Patch',
                        'tooltip': 'If True, a single patch will be processed at a time to save memory.'},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_unet_predictions(viewer: Viewer,
                            image: Image,
                            model_name: str = model_zoo.list_models()[0],
                            dimensionality: str = ALL,
                            modality: str = ALL,
                            output_type: str = ALL,
                            patch_size: Tuple[int, int, int] = (80, 170, 170),
                            patch_halo: Tuple[int, int, int] = (8, 16, 16),
                            single_patch: bool = True,
                            device: str = ALL_DEVICES[0], ) -> Future[LayerDataTuple]:
    out_name = create_layer_name(image.name, model_name)
    inputs_names = (image.name, 'device')

    layer_kwargs = layer_properties(name=out_name,
                                    scale=image.scale,
                                    metadata=image.metadata)

    layer_kwargs['metadata']['pmap'] = True  # this is used to warn the user that the layer is a pmap

    layer_type = 'image'
    step_kwargs = dict(model_name=model_name, patch=patch_size, patch_halo=patch_halo, single_batch_mode=single_patch)

    return start_prediction_process(unet_predictions_wrapper,
                                   runtime_kwargs={'raw': image.data,
                                                   'device': device,
                                                   'handle_multichannel': True},
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='UNet Predictions',
                                   viewer=viewer,
                                   widgets_to_update=[widget_agglomeration.image,
                                                      widget_lifted_multicut.image,
                                                      widget_simple_dt_ws.image,
                                                      widget_split_and_merge_from_scribbles.image]
                                   )


@widget_unet_predictions.image.changed.connect
def _on_widget_unet_predictions_image_change(image: Image):
    _on_prediction_input_image_change(widget_unet_predictions, image)


def _on_any_metadata_changed(modality, output_type, dimensionality):
    modality = [modality] if modality != ALL else None
    output_type = [output_type] if output_type != ALL else None
    dimensionality = [dimensionality] if dimensionality != ALL else None
    widget_unet_predictions.model_name.choices = model_zoo.list_models(
        modality_filter=modality,
        output_type_filter=output_type,
        dimensionality_filter=dimensionality,
    )


widget_unet_predictions.modality.changed.connect(lambda value: _on_any_metadata_changed(value, widget_unet_predictions.output_type.value, widget_unet_predictions.dimensionality.value))
widget_unet_predictions.output_type.changed.connect(lambda value: _on_any_metadata_changed(widget_unet_predictions.modality.value, value, widget_unet_predictions.dimensionality.value))
widget_unet_predictions.dimensionality.changed.connect(lambda value: _on_any_metadata_changed(widget_unet_predictions.modality.value, widget_unet_predictions.output_type.value, value))


@widget_unet_predictions.model_name.changed.connect
def _on_model_name_changed(model_name: str):
    model_name = return_value_if_widget(model_name)
    patch_size = model_zoo.get_model_patch_size(model_name)
    if patch_size is not None:
        widget_unet_predictions.patch_size.value = tuple(patch_size)
    else:
        napari_formatted_logging(f'No recommended patch size for {model_name}',
                                 thread='UNet Predictions',
                                 level='warning')

    description = model_zoo.get_model_description(model_name)
    if description is None:
        description = 'No description available for this model.'
    widget_unet_predictions.model_name.tooltip = f'Select a pretrained model. Current model description: {description}'


def _compute_multiple_predictions(image, patch_size, patch_halo, device, use_custom_models=True):
    out_layers = []
    model_list = model_zoo.list_models(use_custom_models=use_custom_models)
    for i, model_name in enumerate(model_list):

        napari_formatted_logging(f'Running UNet Predictions: {model_name} {i}/{len(model_list)}',
                                 thread='UNet Grid Predictions')

        out_name = create_layer_name(image.name, model_name)
        layer_kwargs = layer_properties(name=out_name,
                                        scale=image.scale,
                                        metadata=image.metadata)
        layer_kwargs['metadata']['pmap'] = True  # this is used to warn the user that the layer is a pmap
        layer_type = 'image'
        try:
            pmap = unet_predictions(raw=image.data, model_name=model_name, patch=patch_size, single_batch_mode=True,
                                    device=device, patch_halo=patch_halo)
            out_layers.append((pmap, layer_kwargs, layer_type))

        except Exception as e:
            print(f'Error while processing: {model_name}')

    return out_layers


@magicgui(call_button='Try all Available Models',
          image={'label': 'Image',
                 'tooltip': 'Raw image to be processed with a neural network.'},
          patch_size={'label': 'Patch size',
                      'tooltip': 'Patch size use to processed the data.'},
          patch_halo={'label': 'Patch halo',
                      'tooltip': 'Patch halo is extra padding for correct prediction on image boarder.'},
          device={'label': 'Device',
                  'choices': ALL_DEVICES},
          use_custom_models={'label': 'Use custom models',
                             'tooltip': 'If True, custom models will also be used.'}
          )
def widget_test_all_unet_predictions(image: Image,
                                     patch_size: Tuple[int, int, int] = (80, 170, 170),
                                     patch_halo: Tuple[int, int, int] = (2, 4, 4),
                                     device: str = ALL_DEVICES[0],
                                     use_custom_models: bool = True) -> Future[List[LayerDataTuple]]:
    func = thread_worker(partial(_compute_multiple_predictions,
                                 image=image,
                                 patch_size=patch_size,
                                 patch_halo=patch_halo,
                                 device=device,
                                 use_custom_models=use_custom_models,))

    future = Future()

    def on_done(result):
        future.set_result(result)

    worker = func()
    worker.returned.connect(on_done)
    worker.start()
    return future


@widget_test_all_unet_predictions.image.changed.connect
def _on_widget_test_all_unet_predictions_image_change(image: Image):
    _on_prediction_input_image_change(widget_test_all_unet_predictions, image)


def _compute_iterative_predictions(pmap, model_name, num_iterations, sigma, patch_size, patch_halo, single_batch_mode, device):
    func = partial(unet_predictions, model_name=model_name, patch=patch_size, patch_halo=patch_halo,
                   single_batch_mode=single_batch_mode, device=device)
    for i in range(num_iterations - 1):
        pmap = func(pmap)
        pmap = image_gaussian_smoothing(image=pmap, sigma=sigma)

    pmap = func(pmap)
    return pmap


@magicgui(call_button='Run Iterative Predictions',
          image={'label': 'Image',
                 'tooltip': 'Raw image to be processed with a neural network.'},
          model_name={'label': 'Select model',
                      'tooltip': f'Select a pretrained model. {model_zoo.list_models()[0]}',
                      'choices': model_zoo.list_models()},
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
          patch_halo={'label': 'Patch halo',
                      'tooltip': 'Patch halo is extra padding for correct prediction on image boarder.'},
          single_patch={'label': 'Single Patch',
                        'tooltip': 'If True, a single patch will be processed at a time to save memory.'},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_iterative_unet_predictions(image: Image,
                                      model_name: str,
                                      num_iterations: int = 2,
                                      sigma: float = 1.0,
                                      patch_size: Tuple[int, int, int] = (80, 170, 170),
                                      patch_halo: Tuple[int, int, int] = (8, 16, 16),
                                      single_patch: bool = True,
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
                       patch_halo=patch_halo,
                       single_batch_mode=single_patch,
                       device=device)

    return start_threading_process(_compute_iterative_predictions,
                                   runtime_kwargs={'pmap': image.data},
                                   statics_kwargs=step_kwargs,
                                   out_name=out_name,
                                   input_keys=inputs_names,
                                   layer_kwarg=layer_kwargs,
                                   layer_type=layer_type,
                                   step_name='UNet Iterative Predictions')


@widget_iterative_unet_predictions.model_name.changed.connect
def _on_model_name_changed_iterative(model_name: str):
    model_name = return_value_if_widget(model_name)
    train_config = model_zoo.get_train_config(model_name)
    patch_size = train_config['loaders']['train']['slice_builder']['patch_shape']
    widget_iterative_unet_predictions.patch_size.value = tuple(patch_size)


@widget_iterative_unet_predictions.image.changed.connect
def _on_widget_iterative_unet_predictions_image_change(image: Image):
    _on_prediction_input_image_change(widget_iterative_unet_predictions, image)


@magicgui(call_button='Add Custom Model',
          new_model_name={'label': 'New model name'},
          model_location={'label': 'Model location',
                          'mode': 'd'},
          resolution={'label': 'Resolution'},
          description={'label': 'Description'},
          dimensionality={'label': 'Dimensionality',
                          'tooltip': 'Dimensionality of the model (2D or 3D). '
                                     'Any 2D model can be used for 3D data.',
                          'widget_type': 'ComboBox',
                          'choices': model_zoo.get_unique_dimensionalities()},
          modality={'label': 'Microscopy modality',
                    'tooltip': 'Modality of the model (e.g. confocal, light-sheet ...).',
                    'widget_type': 'ComboBox',
                    'choices': model_zoo.get_unique_modalities()},
          output_type={'label': 'Prediction type',
                       'widget_type': 'ComboBox',
                       'tooltip': 'Type of prediction (e.g. cell boundaries predictions or nuclei...).',
                       'choices': model_zoo.get_unique_output_types()},

          )
def widget_add_custom_model(new_model_name: str = 'custom_model',
                            model_location: Path = Path.home(),
                            resolution: Tuple[float, float, float] = (1., 1., 1.),
                            description: str = 'New custom model',
                            dimensionality: str = model_zoo.get_unique_dimensionalities()[0],
                            modality: str = model_zoo.get_unique_modalities()[0],
                            output_type: str = model_zoo.get_unique_output_types()[0]) -> None:
    finished, error_msg = model_zoo.add_custom_model(new_model_name=new_model_name,
                                                     location=model_location,
                                                     resolution=resolution,
                                                     description=description,
                                                     dimensionality=dimensionality,
                                                     modality=modality,
                                                     output_type=output_type)

    if finished:
        napari_formatted_logging(f'New model {new_model_name} added to the list of available models.',
                                 level='info',
                                 thread='Add Custom Model')
        widget_unet_predictions.model_name.choices = model_zoo.list_models()
    else:
        napari_formatted_logging(f'Error adding new model {new_model_name} to the list of available models: '
                                 f'{error_msg}',
                                 level='error',
                                 thread='Add Custom Model')
