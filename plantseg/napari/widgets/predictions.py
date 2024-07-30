from concurrent.futures import Future


import torch.cuda
from magicgui import magicgui
from napari import Viewer
from napari.layers import Image
from napari.types import LayerDataTuple
from enum import Enum
from plantseg.models.zoo import model_zoo
from plantseg.napari.logging import napari_formatted_logging
from plantseg.napari.widgets.utils import schedule_task
from plantseg.tasks.predictions_tasks import unet_predictions_task
from plantseg.plantseg_image import PlantSegImage

ALL = 'All'
ALL_CUDA_DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
MPS = ['mps'] if torch.backends.mps.is_available() else []
ALL_DEVICES = ALL_CUDA_DEVICES + MPS + ['cpu']

BIOIMAGEIO_FILTER = [("PlantSeg Only", True), ("All", False)]
SINGLE_PATCH_MODE = [("Auto", False), ("One (lower VRAM usage)", True)]
AUTO_REFRESH_HALO = [("Enable", True), ("Disable", False)]


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
        return [mode.value for mode in cls]


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
    patch_size={'label': 'Patch size', 'tooltip': 'Patch size use to processed the data.'},
    patch_halo={
        'label': 'Patch halo',
        'tooltip': 'Patch halo is extra padding for correct prediction on image boarder.'
        'The value is for one side of a given dimension.',
    },
    recommend_halo={
        'label': 'Recommend halo',
        'tooltip': 'Refresh the halo based on the selected model.',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': AUTO_REFRESH_HALO,
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
    viewer: Viewer,
    image: Image,
    mode: str = UNetPredictionsMode.PLANTSEG.value,
    plantseg_filter: bool = True,
    model_name: str = model_zoo.list_models()[0],
    model_id: str = model_zoo.get_bioimageio_zoo_plantseg_model_names()[0],
    dimensionality: str = ALL,
    modality: str = ALL,
    output_type: str = ALL,
    patch_size: tuple[int, int, int] = (80, 170, 170),
    patch_halo: tuple[int, int, int] = model_zoo.compute_3D_halo_for_zoo_models(model_zoo.list_models()[0]),
    recommend_halo: bool = False,
    single_patch: bool = False,
    device: str = ALL_DEVICES[0],
) -> Future[LayerDataTuple]:
    mode = UNetPredictionsMode(mode)
    if mode == UNetPredictionsMode.PLANTSEG:
        suffix = model_name
        model_id = None
    elif mode == UNetPredictionsMode.BIOIMAGEIO:
        suffix = model_id
        model_name = None

    # TODO add halo support and multichannel support

    ps_image = PlantSegImage.from_napari_layer(image)

    return schedule_task(
        unet_predictions_task,
        task_kwargs={
            "image": ps_image,
            "model_name": model_name,
            "model_id": model_id,
            "suffix": suffix,
            "patch": patch_size,
            "single_batch_mode": single_patch,
            "device": device,
        },
        widget_to_update=[],
    )


def update_halo():
    if widget_unet_predictions.recommend_halo.value:
        napari_formatted_logging(
            'Refreshing halo for the selected model; this might take a while...',
            thread='UNet Predictions',
            level='info',
        )
        if widget_unet_predictions.mode.value == UNetPredictionsMode.PLANTSEG:
            widget_unet_predictions.patch_halo.value = model_zoo.compute_3D_halo_for_zoo_models(
                widget_unet_predictions.model_name.value
            )
        elif widget_unet_predictions.mode.value == UNetPredictionsMode.BIOIMAGEIO:
            widget_unet_predictions.patch_halo.value = model_zoo.compute_3D_halo_for_bioimageio_models(
                widget_unet_predictions.model_id.value
            )
        else:
            raise NotImplementedError(f'Automatic halo not implemented for {widget_unet_predictions.mode.value} mode.')
    else:
        napari_formatted_logging(
            'User selected another model but disabled halo recommendation.', thread='UNet Predictions', level='info'
        )


@widget_unet_predictions.recommend_halo.changed.connect
def _on_widget_unet_predictions_refresh_halo_changed():
    update_halo()


@widget_unet_predictions.mode.changed.connect
def _on_widget_unet_predictions_mode_change(mode: str):
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
    if mode == UNetPredictionsMode.PLANTSEG.value:
        for widget in widgets_p:
            widget.show()
        for widget in widgets_b:
            widget.hide()
    elif mode == UNetPredictionsMode.BIOIMAGEIO.value:
        for widget in widgets_p:
            widget.hide()
        for widget in widgets_b:
            widget.show()
    else:
        raise NotImplementedError(f'Mode {mode} not implemented yet.')
    update_halo()


@widget_unet_predictions.plantseg_filter.changed.connect
def _on_widget_unet_predictions_plantseg_filter_change(plantseg_filter: bool):
    if plantseg_filter:
        widget_unet_predictions.model_id.choices = model_zoo.get_bioimageio_zoo_plantseg_model_names()
    else:
        widget_unet_predictions.model_id.choices = model_zoo.get_bioimageio_zoo_all_model_names()


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
    patch_size = model_zoo.get_model_patch_size(model_name)
    if patch_size is not None:
        widget_unet_predictions.patch_size.value = tuple(patch_size)
    else:
        napari_formatted_logging(
            f'No recommended patch size for {model_name}', thread='UNet Predictions', level='warning'
        )

    description = model_zoo.get_model_description(model_name)
    if description is None:
        description = 'No description available for this model.'
    widget_unet_predictions.model_name.tooltip = f'Select a pretrained model. Current model description: {description}'
    update_halo()


@widget_unet_predictions.model_id.changed.connect
def _on_model_id_changed():
    update_halo()
