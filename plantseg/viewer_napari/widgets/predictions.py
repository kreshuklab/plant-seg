"""UNet Predictions Widget"""

from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Optional

import napari
import torch.cuda
from magicgui import magicgui
from magicgui.types import Separator
from napari.layers import Image
from napari.types import LayerDataTuple

from plantseg.core.image import PlantSegImage
from plantseg.core.zoo import model_zoo
from plantseg.tasks.predictions_tasks import unet_predictions_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.segmentation import widget_agglomeration, widget_dt_ws, widget_lifted_multicut
from plantseg.viewer_napari.widgets.utils import schedule_task

ALL = 'All'
ALL_CUDA_DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
MPS = ['mps'] if torch.backends.mps.is_available() else []
ALL_DEVICES = ALL_CUDA_DEVICES + MPS + ['cpu']

BIOIMAGEIO_FILTER = [("PlantSeg Only", True), ("All", False)]
SINGLE_PATCH_MODE = [("Auto", False), ("One (lower VRAM usage)", True)]
ADVANCED_SETTINGS = [("Enable", True), ("Disable", False)]


########################################################################################################################
#                                                                                                                      #
# UNet Predictions Widget                                                                                              #
#                                                                                                                      #
########################################################################################################################


class UNetPredictionsMode(Enum):
    PLANTSEG = 'PlantSeg Zoo'
    BIOIMAGEIO = 'BioImage.IO Zoo'

    @classmethod
    def to_choices(cls):
        return [(mode.value, mode) for mode in cls]


@magicgui(
    call_button='Run Predictions',
    mode={
        'label': 'Mode',
        'tooltip': 'Select the mode to run the predictions.',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': UNetPredictionsMode.to_choices(),
    },
    image={'label': 'Image', 'tooltip': 'Raw image to be processed with a neural network.'},
    dimensionality={
        'label': 'Dimensionality',
        'tooltip': 'Dimensionality of the model (2D or 3D). '
        'Any 2D model can be used for 3D data. If unsure, select "All".',
        'widget_type': 'ComboBox',
        'choices': [ALL] + model_zoo.get_unique_dimensionalities(),
    },
    modality={
        'label': 'Microscopy modality',
        'tooltip': 'Modality of the model (e.g. confocal, light-sheet ...). If unsure, select "All".',
        'widget_type': 'ComboBox',
        'choices': [ALL] + model_zoo.get_unique_modalities(),
    },
    output_type={
        'label': 'Prediction type',
        'widget_type': 'ComboBox',
        'tooltip': 'Type of prediction (e.g. cell boundaries predictions or nuclei...).' ' If unsure, select "All".',
        'choices': [ALL] + model_zoo.get_unique_output_types(),
    },
    model_name={
        'label': 'PlantSeg model',
        'tooltip': f'Select a pretrained PlantSeg model. '
        f'Current model description: {model_zoo.get_model_description(model_zoo.list_models()[0])}',
        'choices': model_zoo.list_models(),
    },
    model_id={
        'label': 'BioImage.IO model',
        'tooltip': 'Select a model from BioImage.IO model zoo.',
        'choices': model_zoo.get_bioimageio_zoo_plantseg_model_names(),
    },
    plantseg_filter={
        'label': 'Model filter',
        'tooltip': 'Choose to only show models tagged with `plantseg`.',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': BIOIMAGEIO_FILTER,
    },
    advanced={
        'label': 'Show Advanced Parameters',
        'tooltip': 'Change the patch shape, halo shape, and batch size.',
    },
    patch_size={'label': 'Patch size', 'tooltip': 'Patch size used to process the data.'},
    patch_halo={
        'label': 'Patch halo',
        'tooltip': 'Patch halo is extra padding for correct prediction on image borders. '
        'The value is for one side of a given dimension.',
    },
    single_patch={
        'label': 'Batch size',
        'tooltip': 'Single patch = batch size 1 (lower GPU memory usage);\nFind Batch Size = find the biggest batch size.',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': SINGLE_PATCH_MODE,
    },
    device={'label': 'Device', 'choices': ALL_DEVICES},
)
def widget_unet_predictions(
    viewer: napari.Viewer,
    image: Image,
    mode: UNetPredictionsMode = UNetPredictionsMode.PLANTSEG,
    plantseg_filter: bool = True,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    dimensionality: str = ALL,
    modality: str = ALL,
    output_type: str = ALL,
    device: str = ALL_DEVICES[0],
    advanced: bool = False,
    patch_size: tuple[int, int, int] = (80, 170, 170),
    patch_halo: tuple[int, int, int] = (0, 0, 0),
    single_patch: bool = False,
) -> Future[list[LayerDataTuple]]:
    if mode is UNetPredictionsMode.PLANTSEG:
        suffix = model_name
        model_id = None
    elif mode is UNetPredictionsMode.BIOIMAGEIO:
        suffix = model_id
        model_name = None
    else:
        raise NotImplementedError(f'Mode {mode} not implemented yet.')

    ps_image = PlantSegImage.from_napari_layer(image)
    return schedule_task(
        unet_predictions_task,
        task_kwargs={
            "image": ps_image,
            "model_name": model_name,
            "model_id": model_id,
            "suffix": suffix,
            "patch": patch_size if advanced else None,
            "patch_halo": patch_halo if advanced else None,
            "single_batch_mode": single_patch if advanced else False,
            "device": device,
        },
        widgets_to_update=[
            widget_dt_ws.image,
            widget_agglomeration.image,
            widget_lifted_multicut.image,
        ],
    )


advanced_unet_predictions_widgets = [
    widget_unet_predictions.patch_size,
    widget_unet_predictions.patch_halo,
    widget_unet_predictions.single_patch,
]
[widget.hide() for widget in advanced_unet_predictions_widgets]


def update_halo():
    if widget_unet_predictions.advanced.value:
        log(
            'Refreshing halo for the selected model; this might take a while...',
            thread='UNet predictions',
            level='info',
        )
        if widget_unet_predictions.mode.value is UNetPredictionsMode.PLANTSEG:
            widget_unet_predictions.patch_halo.value = model_zoo.compute_3D_halo_for_zoo_models(
                widget_unet_predictions.model_name.value
            )
        elif widget_unet_predictions.mode.value is UNetPredictionsMode.BIOIMAGEIO:
            widget_unet_predictions.patch_halo.value = model_zoo.compute_3D_halo_for_bioimageio_models(
                widget_unet_predictions.model_id.value
            )
        else:
            raise NotImplementedError(f'Automatic halo not implemented for {widget_unet_predictions.mode.value} mode.')


@widget_unet_predictions.advanced.changed.connect
def _on_widget_unet_predictions_advanced_changed(advanced):
    if advanced:
        update_halo()
        [widget.show() for widget in advanced_unet_predictions_widgets]
    else:
        [widget.hide() for widget in advanced_unet_predictions_widgets]


@widget_unet_predictions.mode.changed.connect
def _on_widget_unet_predictions_mode_change(mode: UNetPredictionsMode):
    widgets_p = [
        widget_unet_predictions.model_name,
        widget_unet_predictions.dimensionality,
        widget_unet_predictions.modality,
        widget_unet_predictions.output_type,
    ]
    widgets_b = [
        widget_unet_predictions.model_id,
        widget_unet_predictions.plantseg_filter,
    ]
    if mode is UNetPredictionsMode.PLANTSEG:
        for widget in widgets_p:
            widget.show()
        for widget in widgets_b:
            widget.hide()
    elif mode is UNetPredictionsMode.BIOIMAGEIO:
        for widget in widgets_p:
            widget.hide()
        for widget in widgets_b:
            widget.show()
    else:
        raise NotImplementedError(f'Mode {mode} not implemented yet.')

    if widget_unet_predictions.advanced.value:
        update_halo()


@widget_unet_predictions.plantseg_filter.changed.connect
def _on_widget_unet_predictions_plantseg_filter_change(plantseg_filter: bool):
    if plantseg_filter:
        widget_unet_predictions.model_id.choices = model_zoo.get_bioimageio_zoo_plantseg_model_names()
    else:
        widget_unet_predictions.model_id.choices = (
            model_zoo.get_bioimageio_zoo_plantseg_model_names()
            + [Separator]
            + model_zoo.get_bioimageio_zoo_other_model_names()
        )


# TODO reinsert this code when _on_prediction_input_image_change is implemented
# @widget_unet_predictions.image.changed.connect
# def _on_widget_unet_predictions_image_change(image: Image):
#     _on_prediction_input_image_change(widget_unet_predictions, image)


def _on_any_metadata_changed(modality, output_type, dimensionality):
    modality = [modality] if modality != ALL else None
    output_type = [output_type] if output_type != ALL else None
    dimensionality = [dimensionality] if dimensionality != ALL else None
    widget_unet_predictions.model_name.choices = model_zoo.list_models(
        modality_filter=modality,
        output_type_filter=output_type,
        dimensionality_filter=dimensionality,
    )


widget_unet_predictions.modality.changed.connect(
    lambda value: _on_any_metadata_changed(
        value, widget_unet_predictions.output_type.value, widget_unet_predictions.dimensionality.value
    )
)
widget_unet_predictions.output_type.changed.connect(
    lambda value: _on_any_metadata_changed(
        widget_unet_predictions.modality.value, value, widget_unet_predictions.dimensionality.value
    )
)
widget_unet_predictions.dimensionality.changed.connect(
    lambda value: _on_any_metadata_changed(
        widget_unet_predictions.modality.value, widget_unet_predictions.output_type.value, value
    )
)


@widget_unet_predictions.model_name.changed.connect
def _on_model_name_changed(model_name: str):
    description = model_zoo.get_model_description(model_name)
    if description is None:
        description = 'No description available for this model.'
    widget_unet_predictions.model_name.tooltip = f'Select a pretrained model. Current model description: {description}'
    if widget_unet_predictions.advanced.value:
        update_halo()


@widget_unet_predictions.model_id.changed.connect
def _on_model_id_changed(model_id: str):
    if widget_unet_predictions.advanced.value:
        update_halo()


########################################################################################################################
#                                                                                                                      #
# Add Custom Model Widget                                                                                              #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button='Add Custom Model',
    new_model_name={'label': 'New model name'},
    model_location={'label': 'Model location', 'mode': 'd'},
    resolution={'label': 'Resolution', 'options': {'step': 0.00001}},
    description={'label': 'Description'},
    dimensionality={
        'label': 'Dimensionality',
        'tooltip': 'Dimensionality of the model (2D or 3D). Any 2D model can be used for 3D data.',
        'widget_type': 'ComboBox',
        'choices': model_zoo.get_unique_dimensionalities(),
    },
    modality={
        'label': 'Microscopy modality',
        'tooltip': 'Modality of the model (e.g. confocal, light-sheet ...).',
        'widget_type': 'ComboBox',
        'choices': model_zoo.get_unique_modalities(),
    },
    output_type={
        'label': 'Prediction type',
        'widget_type': 'ComboBox',
        'tooltip': 'Type of prediction (e.g. cell boundaries predictions or nuclei...).',
        'choices': model_zoo.get_unique_output_types(),
    },
)
def widget_add_custom_model(
    new_model_name: str = 'custom_model',
    model_location: Path = Path.home(),
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    description: str = 'A model trained by the user.',
    dimensionality: str = model_zoo.get_unique_dimensionalities()[0],
    modality: str = model_zoo.get_unique_modalities()[0],
    output_type: str = model_zoo.get_unique_output_types()[0],
) -> None:
    finished, error_msg = model_zoo.add_custom_model(
        new_model_name=new_model_name,
        location=model_location,
        resolution=resolution,
        description=description,
        dimensionality=dimensionality,
        modality=modality,
        output_type=output_type,
    )

    if finished:
        log(
            f'New model {new_model_name} added to the list of available models.',
            level='info',
            thread='Add Custom Model',
        )
        widget_unet_predictions.model_name.choices = model_zoo.list_models()
    else:
        log(
            f'Error adding new model {new_model_name} to the list of available models: ' f'{error_msg}',
            level='error',
            thread='Add Custom Model',
        )
