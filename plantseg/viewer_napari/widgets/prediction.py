"""UNet Prediction Widget"""

from enum import Enum
from pathlib import Path
from typing import Optional

import torch.cuda
from magicgui import magicgui
from magicgui.types import Separator
from magicgui.widgets import Container, ProgressBar, create_widget
from napari.layers import Image

from plantseg.core.image import PlantSegImage
from plantseg.core.zoo import model_zoo
from plantseg.tasks.prediction_tasks import biio_prediction_task, unet_prediction_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.proofreading import (
    widget_split_and_merge_from_scribbles,
)
from plantseg.viewer_napari.widgets.segmentation import (
    widget_agglomeration,
    widget_dt_ws,
)
from plantseg.viewer_napari.widgets.utils import schedule_task

ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
MPS = ["mps"] if torch.backends.mps.is_available() else []
ALL_DEVICES = ALL_CUDA_DEVICES + MPS + ["cpu"]

BIOIMAGEIO_FILTER = [("PlantSeg Only", True), ("All", False)]
SINGLE_PATCH_MODE = [("Auto", False), ("One (lower VRAM usage)", True)]
ADVANCED_SETTINGS = [("Enable", True), ("Disable", False)]

# Using Enum causes more complexity, stay constant
ALL_DIMENSIONS = "All dimensions"
ALL_MODALITIES = "All modalities"
ALL_TYPES = "All types"
CUSTOM = "Custom"


########################################################################################################################
#                                                                                                                      #
# UNet Prediction Widget                                                                                               #
#                                                                                                                      #
########################################################################################################################


class UNetPredictionMode(Enum):
    PLANTSEG = "PlantSeg Zoo"
    BIOIMAGEIO = "BioImage.IO Zoo"

    @classmethod
    def to_choices(cls):
        return [(mode.value, mode) for mode in cls]


model_filters = Container(
    widgets=[
        create_widget(
            annotation=str,
            name="dimensionality",
            label="Dimensionality",
            widget_type="ComboBox",
            options={
                "choices": [ALL_DIMENSIONS] + model_zoo.get_unique_dimensionalities()
            },
        ),
        create_widget(
            annotation=str,
            name="modality",
            label="Microscopy modality",
            widget_type="ComboBox",
            options={"choices": [ALL_MODALITIES] + model_zoo.get_unique_modalities()},
        ),
        create_widget(
            annotation=str,
            name="output_type",
            label="Prediction type",
            widget_type="ComboBox",
            options={"choices": [ALL_TYPES] + model_zoo.get_unique_output_types()},
        ),
    ],
    label="Model filters",
    layout="horizontal",
    labels=False,
)


@magicgui(
    call_button="Image to Prediction",
    mode={
        "label": "Mode",
        "tooltip": "Select the mode to run the prediction.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": UNetPredictionMode.to_choices(),
    },
    image={
        "label": "Image",
        "tooltip": "Raw image to be processed with a neural network.",
    },
    plantseg_filter={
        "label": "Model filter",
        "tooltip": "Choose to only show models tagged with `plantseg`.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": BIOIMAGEIO_FILTER,
    },
    model_name={
        "label": "PlantSeg model",
        "tooltip": f"Select a pretrained PlantSeg model. "
        f"Current model description: {model_zoo.get_model_description(model_zoo.list_models()[0])}",
        "choices": model_zoo.list_models(),
    },
    model_id={
        "label": "BioImage.IO model",
        "tooltip": "Select a model from BioImage.IO model zoo.",
        "choices": model_zoo.get_bioimageio_zoo_plantseg_model_names(),
        "value": model_zoo.get_bioimageio_zoo_plantseg_model_names()[0][1],
    },
    advanced={
        "label": "Show advanced parameters",
        "tooltip": "Change the patch shape, halo shape, and batch size.",
    },
    patch_size={
        "label": "Patch size",
        "tooltip": "Patch size used to process the data.",
    },
    patch_halo={
        "label": "Patch halo",
        "tooltip": "Patch halo is extra padding for correct prediction on image borders. "
        "The value is for one side of a given dimension.",
    },
    single_patch={
        "label": "Batch size",
        "tooltip": "Single patch = batch size 1 (lower GPU memory usage);\nFind Batch Size = find the biggest batch size.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": SINGLE_PATCH_MODE,
    },
    device={"label": "Device", "choices": ALL_DEVICES},
    pbar={"label": "Progress", "max": 0, "min": 0, "visible": False},
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_unet_prediction(
    image: Image,
    mode: UNetPredictionMode = UNetPredictionMode.PLANTSEG,
    plantseg_filter: bool = True,
    model_name: Optional[str] = None,
    model_id: Optional[str] = model_zoo.get_bioimageio_zoo_plantseg_model_names()[0][1],
    device: str = ALL_DEVICES[0],
    advanced: bool = False,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    patch_halo: tuple[int, int, int] = (0, 0, 0),
    single_patch: bool = False,
    pbar: Optional[ProgressBar] = None,
    update_other_widgets: bool = True,
) -> None:
    ps_image = PlantSegImage.from_napari_layer(image)

    if mode is UNetPredictionMode.PLANTSEG:
        suffix = model_name
        model_id = None
        widgets_to_update = [
            widget_dt_ws.image,
            widget_agglomeration.image,
            widget_split_and_merge_from_scribbles.image,
        ]
        return schedule_task(
            unet_prediction_task,
            task_kwargs={
                "image": ps_image,
                "model_name": model_name,
                "model_id": model_id,
                "suffix": suffix,
                "patch": patch_size if advanced else None,
                "patch_halo": patch_halo if advanced else None,
                "single_batch_mode": single_patch if advanced else False,
                "device": device,
                "_pbar": pbar,
                "_to_hide": [widget_unet_prediction.call_button],
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )
    elif mode is UNetPredictionMode.BIOIMAGEIO:
        suffix = model_id
        model_name = None
        widgets_to_update = [
            # BioImage.IO models may output multi-channel 3D image or even multi-channel scalar in CZYX format.
            # So PlantSeg widgets, which all take ZYX or YX, are better not to be updated.
        ]
        return schedule_task(
            biio_prediction_task,
            task_kwargs={
                "image": ps_image,
                "model_id": model_id,
                "suffix": suffix,
                "_pbar": pbar,
                "_to_hide": [widget_unet_prediction.call_button],
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )
    else:
        raise NotImplementedError(f"Mode {mode} not implemented yet.")


widget_unet_prediction.insert(3, model_filters)

advanced_unet_prediction_widgets = [
    widget_unet_prediction.patch_size,
    widget_unet_prediction.patch_halo,
    widget_unet_prediction.single_patch,
]
[widget.hide() for widget in advanced_unet_prediction_widgets]


def update_halo():
    if (
        widget_unet_prediction.model_name.value is None
        and widget_unet_prediction.model_id.value is None
    ):
        return
    if widget_unet_prediction.advanced.value:
        log(
            "Refreshing halo for the selected model; this might take a while...",
            thread="UNet prediction",
            level="info",
        )

        if widget_unet_prediction.mode.value is UNetPredictionMode.PLANTSEG:
            widget_unet_prediction.patch_halo.value = (
                model_zoo.compute_3D_halo_for_zoo_models(
                    widget_unet_prediction.model_name.value
                )
            )
            if model_zoo.is_2D_zoo_model(widget_unet_prediction.model_name.value):
                widget_unet_prediction.patch_size[0].value = 1
                widget_unet_prediction.patch_size[0].enabled = False
                widget_unet_prediction.patch_halo[0].enabled = False
            else:
                widget_unet_prediction.patch_size[
                    0
                ].value = widget_unet_prediction.patch_size[1].value
                widget_unet_prediction.patch_size[0].enabled = True
                widget_unet_prediction.patch_halo[0].enabled = True
        elif widget_unet_prediction.mode.value is UNetPredictionMode.BIOIMAGEIO:
            log(
                "Automatic halo not implemented for BioImage.IO models yet because they are handled by BioImage.IO Core.",
                thread="BioImage.IO Core prediction",
                level="info",
            )
        else:
            raise NotImplementedError(
                f"Automatic halo not implemented for {widget_unet_prediction.mode.value} mode."
            )


@widget_unet_prediction.advanced.changed.connect
def _on_widget_unet_prediction_advanced_changed(advanced):
    if advanced:
        update_halo()
        for widget in advanced_unet_prediction_widgets:
            widget.show()
    else:
        for widget in advanced_unet_prediction_widgets:
            widget.hide()


@widget_unet_prediction.mode.changed.connect
def _on_widget_unet_prediction_mode_change(mode: UNetPredictionMode):
    widgets_p = [  # PlantSeg
        widget_unet_prediction.model_name,
        model_filters,
    ]
    widgets_b = [  # BioImage.IO
        widget_unet_prediction.model_id,
        widget_unet_prediction.plantseg_filter,
    ]
    if mode is UNetPredictionMode.PLANTSEG:
        for widget in widgets_p:
            widget.show()
        for widget in widgets_b:
            widget.hide()
    elif mode is UNetPredictionMode.BIOIMAGEIO:
        for widget in widgets_p:
            widget.hide()
        for widget in widgets_b:
            widget.show()
    else:
        raise NotImplementedError(f"Mode {mode} not implemented yet.")

    if widget_unet_prediction.advanced.value:
        update_halo()


@widget_unet_prediction.plantseg_filter.changed.connect
def _on_widget_unet_prediction_plantseg_filter_change(plantseg_filter: bool):
    if plantseg_filter:
        widget_unet_prediction.model_id.choices = (
            model_zoo.get_bioimageio_zoo_plantseg_model_names()
        )
    else:
        widget_unet_prediction.model_id.choices = (
            model_zoo.get_bioimageio_zoo_plantseg_model_names()
            + [
                ("", Separator)
            ]  # `[('', Separator)]` for list[tuple[str, str]], [Separator] for list[str]
            + model_zoo.get_bioimageio_zoo_other_model_names()
        )


# TODO reinsert this code when _on_prediction_input_image_change is implemented
# @widget_unet_prediction.image.changed.connect
# def _on_widget_unet_prediction_image_change(image: Image):
#     _on_prediction_input_image_change(widget_unet_prediction, image)


@model_filters.changed.connect
def _on_any_metadata_changed(widget):
    modality = widget.modality.value
    output_type = widget.output_type.value
    dimensionality = widget.dimensionality.value

    modality = [modality] if modality != ALL_MODALITIES else None
    output_type = [output_type] if output_type != ALL_TYPES else None
    dimensionality = [dimensionality] if dimensionality != ALL_DIMENSIONS else None
    widget_unet_prediction.model_name.choices = model_zoo.list_models(
        modality_filter=modality,
        output_type_filter=output_type,
        dimensionality_filter=dimensionality,
    )


@widget_unet_prediction.model_name.changed.connect
def _on_model_name_changed(model_name: str):
    description = model_zoo.get_model_description(model_name)
    if description is None:
        description = "No description available for this model."
        widget_unet_prediction.advanced.hide()
        widget_unet_prediction.device.hide()
    else:
        widget_unet_prediction.advanced.show()
        widget_unet_prediction.device.show()
    widget_unet_prediction.model_name.tooltip = (
        f"Select a pretrained model. Current model description: {description}"
    )

    if widget_unet_prediction.advanced.value:
        update_halo()


widget_unet_prediction.advanced.hide()
widget_unet_prediction.device.hide()


@widget_unet_prediction.model_id.changed.connect
def _on_model_id_changed(model_id: str):
    if widget_unet_prediction.advanced.value:
        update_halo()


########################################################################################################################
#                                                                                                                      #
# Add Custom Model Widget                                                                                              #
#                                                                                                                      #
########################################################################################################################
@magicgui(call_button="Add Custom Model")
def widget_add_custom_model_toggl() -> None:
    widget_unet_prediction.hide()
    widget_add_custom_model_toggl.hide()
    widget_add_custom_model.show()


@magicgui(
    call_button="Add Custom Model",
    new_model_name={"label": "New model name"},
    model_location={"label": "Model location", "mode": "d"},
    resolution={
        "label": "Voxel Size",
        "options": {"step": 0.00001},
        "tooltip": "Resolution of the dataset used to model in micrometers per pixel.",
    },
    description={"label": "Description"},
    dimensionality={
        "label": "Dimensionality",
        "tooltip": "Dimensionality of the model (2D or 3D). Any 2D model can be used for 3D data.",
        "widget_type": "ComboBox",
        "choices": model_zoo.get_unique_dimensionalities(),
    },
    modality={
        "label": "Microscopy modality",
        "tooltip": "Modality of the model (e.g. confocal, light-sheet ...).",
        "widget_type": "ComboBox",
        "choices": model_zoo.get_unique_modalities() + [CUSTOM],
    },
    custom_modality={"label": "Custom modality"},
    output_type={
        "label": "Prediction type",
        "widget_type": "ComboBox",
        "tooltip": "Type of prediction (e.g. cell boundaries prediction or nuclei...).",
        "choices": model_zoo.get_unique_output_types() + [CUSTOM],
    },
    custom_output_type={"label": "Custom type"},
)
def widget_add_custom_model(
    new_model_name: str = "custom_model",
    model_location: Path = Path.home(),
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    description: str = "A model trained by the user.",
    dimensionality: str = model_zoo.get_unique_dimensionalities()[0],
    modality: str = model_zoo.get_unique_modalities()[0],
    custom_modality: str = "",
    output_type: str = model_zoo.get_unique_output_types()[0],
    custom_output_type: str = "",
) -> None:
    if modality == CUSTOM:
        modality = custom_modality
    if output_type == CUSTOM:
        output_type = custom_output_type

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
            f"New model {new_model_name} added to the list of available models.",
            level="info",
            thread="Add Custom Model",
        )
        widget_unet_prediction.model_name.choices = model_zoo.list_models()
    else:
        log(
            f"Error adding new model {new_model_name} to the list of available models: {error_msg}",
            level="error",
            thread="Add Custom Model",
        )
    widget_add_custom_model.hide()
    widget_unet_prediction.show()
    widget_add_custom_model_toggl.show()


widget_add_custom_model.hide()
widget_add_custom_model.custom_modality.hide()
widget_add_custom_model.custom_output_type.hide()


@widget_add_custom_model.modality.changed.connect
def _on_custom_modality_change(modality: str):
    if modality == CUSTOM:
        widget_add_custom_model.custom_modality.show()
    else:
        widget_add_custom_model.custom_modality.hide()


@widget_add_custom_model.output_type.changed.connect
def _on_custom_output_type_change(output_type: str):
    if output_type == CUSTOM:
        widget_add_custom_model.custom_output_type.show()
    else:
        widget_add_custom_model.custom_output_type.hide()
