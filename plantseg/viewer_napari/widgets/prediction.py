"""UNet Prediction Widget"""

from enum import Enum
from pathlib import Path
from typing import Optional

import torch.cuda
from magicgui import magic_factory
from magicgui.types import Separator, Undefined
from magicgui.widgets import ComboBox, Container, ProgressBar

from plantseg import logger
from plantseg.core.image import PlantSegImage
from plantseg.core.zoo import model_zoo
from plantseg.tasks.prediction_tasks import biio_prediction_task, unet_prediction_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.utils import schedule_task


class UNetPredictionMode(Enum):
    PLANTSEG = "PlantSeg Zoo"
    BIOIMAGEIO = "BioImage.IO Zoo"

    @classmethod
    def to_choices(cls):
        return [(mode.value, mode) for mode in cls]


class Prediction_Widgets:
    def __init__(self, widget_layer_select):
        self.widget_layer_select = widget_layer_select
        # Constants
        self.ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.MPS = ["mps"] if torch.backends.mps.is_available() else []
        self.ALL_DEVICES = self.ALL_CUDA_DEVICES + self.MPS + ["cpu"]

        self.BIOIMAGEIO_FILTER = [("PlantSeg Only", True), ("All", False)]
        self.SINGLE_PATCH_MODE = [("Auto", False), ("One (lower VRAM usage)", True)]
        self.ADVANCED_SETTINGS = [("Enable", True), ("Disable", False)]

        self.ALL_DIMENSIONS = "All dimensions"
        self.ALL_MODALITIES = "All modalities"
        self.ALL_TYPES = "All types"
        self.CUSTOM = "Custom"
        self.ADD_MODEL = "ADD CUSTOM MODEL"

        # @@@@@ model filter container @@@@@
        self.model_filters = Container(
            widgets=[
                ComboBox(
                    choices=[self.ALL_DIMENSIONS]
                    + model_zoo.get_unique_dimensionalities(),
                    label="Dimensionality",
                    name="dimensionality",
                ),
                ComboBox(
                    name="modality",
                    label="Microscopy modality",
                    choices=[self.ALL_MODALITIES] + model_zoo.get_unique_modalities(),
                ),
                ComboBox(
                    name="output_type",
                    label="Prediction type",
                    choices=[self.ALL_TYPES] + model_zoo.get_unique_output_types(),
                ),
            ],
            label="Model filters",
            layout="horizontal",
            labels=False,
        )
        self.model_filters.changed.connect(self._on_any_metadata_changed)

        # @@@@@ Unet Prediction @@@@@
        self.widget_unet_prediction = self.factory_unet_prediction()
        self.widget_unet_prediction.self.bind(self)
        self.widget_unet_prediction.model_name._default_choices = (
            lambda _: model_zoo.list_models() + [self.ADD_MODEL]
        )
        self.widget_unet_prediction.model_name.reset_choices()
        self.widget_unet_prediction.plantseg_filter._default_choices = (
            self.BIOIMAGEIO_FILTER
        )
        self.widget_unet_prediction.plantseg_filter.reset_choices()
        self.widget_unet_prediction.plantseg_filter.value = True

        self.widget_unet_prediction.single_patch._default_choices = (
            self.SINGLE_PATCH_MODE
        )
        self.widget_unet_prediction.single_patch.reset_choices()
        self.widget_unet_prediction.single_patch.value = False
        self.widget_unet_prediction.device._default_choices = self.ALL_DEVICES
        self.widget_unet_prediction.device.reset_choices()
        self.widget_unet_prediction.device.value = self.ALL_DEVICES[0]

        self.widget_unet_prediction.insert(3, self.model_filters)

        self.advanced_unet_prediction_widgets = [
            self.widget_unet_prediction.patch_size,
            self.widget_unet_prediction.patch_halo,
            self.widget_unet_prediction.single_patch,
        ]
        [widget.hide() for widget in self.advanced_unet_prediction_widgets]

        self.widget_unet_prediction.advanced.changed.connect(
            self._on_widget_unet_prediction_advanced_changed
        )
        self.widget_unet_prediction.mode.changed.connect(
            self._on_widget_unet_prediction_mode_change
        )
        self.widget_unet_prediction.plantseg_filter.changed.connect(
            self._on_widget_unet_prediction_plantseg_filter_change
        )

        self.widget_unet_prediction.model_name.changed.connect(
            self._on_model_name_changed
        )

        self.widget_unet_prediction.model_id.changed.connect(self._on_model_id_changed)

        # @@@@@ custom model @@@@@
        #
        self.widget_add_custom_model = self.factory_add_custom_model()
        self.widget_add_custom_model.self.bind(self)
        self.widget_add_custom_model.cancel_button.changed.connect(
            self.cancel_custom_model
        )
        self.widget_add_custom_model.hide()
        self.widget_add_custom_model.custom_modality.hide()
        self.widget_add_custom_model.custom_output_type.hide()

        self.widget_add_custom_model.modality.changed.connect(
            self._on_custom_modality_change
        )
        self.widget_add_custom_model.output_type.changed.connect(
            self._on_custom_output_type_change
        )
        self.widget_add_custom_model.modality._default_choices = (
            model_zoo.get_unique_modalities() + [self.CUSTOM]
        )
        self.widget_add_custom_model.modality.reset_choices()
        self.widget_add_custom_model.output_type._default_choices = (
            model_zoo.get_unique_output_types() + [self.CUSTOM]
        )
        self.widget_add_custom_model.output_type.reset_choices()

        self.widget_add_custom_model_toggle = self.factory_add_custom_model_toggle()
        self.widget_add_custom_model_toggle.self.bind(self)
        logger.debug(f"End init device: {self.widget_unet_prediction.device.value}")

    @magic_factory(
        call_button="Image to Prediction",
        mode={
            "label": "Mode",
            "tooltip": "Select the mode to run the prediction.",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": UNetPredictionMode.to_choices(),
            "value": UNetPredictionMode.PLANTSEG,
        },
        plantseg_filter={
            "label": "Model filter",
            "tooltip": "Choose to only show models tagged with `plantseg`.",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        model_name={
            "label": "PlantSeg model",
            "tooltip": f"Select a pretrained PlantSeg model. "
            f"Current model description: {model_zoo.get_model_description(model_zoo.list_models()[0])}",
            "choices": [None],
            "value": None,
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
            "choices": [True, False],
        },
        device={
            "label": "Device",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        pbar={"label": "Progress", "max": 0, "min": 0, "visible": False},
        update_other_widgets={
            "visible": False,
            "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
        },
    )
    def factory_unet_prediction(
        self,
        mode: UNetPredictionMode,
        plantseg_filter: bool,
        model_name: Optional[str],
        model_id: Optional[str],
        device: str,
        advanced: bool = False,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        patch_halo: tuple[int, int, int] = (0, 0, 0),
        single_patch: bool = False,
        pbar: Optional[ProgressBar] = None,
        update_other_widgets: bool = True,
    ) -> None:
        if self.widget_layer_select.layer.value is None:
            log("Please load an image first!", thread="Prediction", level="WARNING")
            return
        ps_image = PlantSegImage.from_napari_layer(self.widget_layer_select.layer.value)

        if mode is UNetPredictionMode.PLANTSEG:
            if model_name is None:
                log(
                    "Choose a model first!",
                    thread="Prediction",
                    level="WARNING",
                )
                return
            suffix = model_name
            model_id = None
            widgets_to_update = []
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
                    "_to_hide": [self.widget_unet_prediction.call_button],
                },
                widgets_to_update=widgets_to_update if update_other_widgets else [],
            )
        elif mode is UNetPredictionMode.BIOIMAGEIO:
            if model_id is None:
                return
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
                    "_to_hide": [self.widget_unet_prediction.call_button],
                },
                widgets_to_update=widgets_to_update if update_other_widgets else [],
            )
        else:
            raise NotImplementedError(f"Mode {mode} not implemented yet.")

    def update_halo(self):
        if self.widget_unet_prediction.advanced.value:
            log(
                "Refreshing halo for the selected model; this might take a while...",
                thread="UNet prediction",
                level="info",
            )

            if self.widget_unet_prediction.mode.value is UNetPredictionMode.PLANTSEG:
                if self.widget_unet_prediction.model_name.value is None:
                    return
                self.widget_unet_prediction.patch_halo.value = (
                    model_zoo.compute_3D_halo_for_zoo_models(
                        self.widget_unet_prediction.model_name.value
                    )
                )
                if model_zoo.is_2D_zoo_model(
                    self.widget_unet_prediction.model_name.value
                ):
                    self.widget_unet_prediction.patch_size[0].value = 1
                    self.widget_unet_prediction.patch_size[0].enabled = False
                    self.widget_unet_prediction.patch_halo[0].enabled = False
                else:
                    if self.widget_unet_prediction.model_id.value is None:
                        return
                    self.widget_unet_prediction.patch_size[
                        0
                    ].value = self.widget_unet_prediction.patch_size[1].value
                    self.widget_unet_prediction.patch_size[0].enabled = True
                    self.widget_unet_prediction.patch_halo[0].enabled = True
            elif (
                self.widget_unet_prediction.mode.value is UNetPredictionMode.BIOIMAGEIO
            ):
                log(
                    "Automatic halo not implemented for BioImage.IO models yet because they are handled by BioImage.IO Core.",
                    thread="BioImage.IO Core prediction",
                    level="info",
                )
            else:
                raise NotImplementedError(
                    f"Automatic halo not implemented for {self.widget_unet_prediction.mode.value} mode."
                )

    def _on_widget_unet_prediction_advanced_changed(self, advanced):
        logger.debug(f"_on_widget_unet_prediction_advanced_changed called: {advanced}")
        if advanced:
            self.update_halo()
            for widget in self.advanced_unet_prediction_widgets:
                widget.show()
        else:
            for widget in self.advanced_unet_prediction_widgets:
                widget.hide()

    def _on_widget_unet_prediction_mode_change(self, mode: UNetPredictionMode):
        logger.debug(f"_on_widget_unet_prediction_mode_change called: {mode}")
        widgets_p = [  # PlantSeg
            self.widget_unet_prediction.model_name,
            self.model_filters,
        ]
        widgets_b = [  # BioImage.IO
            self.widget_unet_prediction.model_id,
            self.widget_unet_prediction.plantseg_filter,
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

        if self.widget_unet_prediction.advanced.value:
            self.update_halo()

    def _on_widget_unet_prediction_plantseg_filter_change(self, plantseg_filter: bool):
        logger.debug(
            f"_on_widget_unet_prediction_plantseg_filter_change called: {plantseg_filter}"
        )
        if plantseg_filter:
            self.widget_unet_prediction.model_id.choices = (
                model_zoo.get_bioimageio_zoo_plantseg_model_names()
            )
        else:
            self.widget_unet_prediction.model_id.choices = (
                model_zoo.get_bioimageio_zoo_plantseg_model_names()
                + [
                    ("", Separator)
                ]  # `[('', Separator)]` for list[tuple[str, str]], [Separator] for list[str]
                + model_zoo.get_bioimageio_zoo_other_model_names()
            )

    def _on_any_metadata_changed(self, widget):
        logger.debug("_on_any_metadata_changed called!")
        modality = widget.modality.value
        output_type = widget.output_type.value
        dimensionality = widget.dimensionality.value

        modality = [modality] if modality != self.ALL_MODALITIES else None
        output_type = [output_type] if output_type != self.ALL_TYPES else None
        dimensionality = (
            [dimensionality] if dimensionality != self.ALL_DIMENSIONS else None
        )
        self.widget_unet_prediction.model_name.choices = model_zoo.list_models(
            modality_filter=modality,
            output_type_filter=output_type,
            dimensionality_filter=dimensionality,
        ) + [self.ADD_MODEL]

    def _on_model_name_changed(self, model_name: str | None):
        logger.debug(f"_on_model_name_changed called: {model_name}")
        if model_name is None:
            return
        elif model_name == self.ADD_MODEL:
            self.widget_add_custom_model_toggle()
            return

        description = model_zoo.get_model_description(model_name)
        if description is None:
            description = "No description available for this model."
            self.widget_unet_prediction.advanced.hide()
            self.widget_unet_prediction.device.hide()
        else:
            self.widget_unet_prediction.advanced.show()
            self.widget_unet_prediction.device.show()
        self.widget_unet_prediction.model_name.tooltip = (
            f"Select a pretrained model. Current model description: {description}"
        )

        if self.widget_unet_prediction.advanced.value:
            self.update_halo()

    def _on_model_id_changed(self, model_id: str):
        if self.widget_unet_prediction.advanced.value:
            self.update_halo()

    # TODO: Make option in drop-down
    @magic_factory(call_button="Add Custom Model")
    def factory_add_custom_model_toggle(self) -> None:
        self.widget_unet_prediction.hide()
        self.widget_add_custom_model_toggle.hide()
        self.widget_add_custom_model.show()

    @magic_factory(
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
            "value": Undefined,
        },
        modality={
            "label": "Microscopy modality",
            "tooltip": "Modality of the model (e.g. confocal, light-sheet ...).",
            "widget_type": "ComboBox",
            "value": Undefined,
        },
        custom_modality={"label": "Custom modality", "value": Undefined},
        output_type={
            "label": "Prediction type",
            "widget_type": "ComboBox",
            "tooltip": "Type of prediction (e.g. cell boundaries prediction or nuclei...).",
            "value": Undefined,
        },
        custom_output_type={"label": "Custom type", "value": Undefined},
        cancel_button={"label": "Cancel", "widget_type": "PushButton"},
    )
    def factory_add_custom_model(
        self,
        new_model_name: str = "custom_model",
        model_location: Path = Path.home(),
        resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
        description: str = "A model trained by the user.",
        dimensionality: str = "",
        modality: str = "",
        custom_modality: str = "",
        output_type: str = "",
        custom_output_type: str = "",
        cancel_button: bool = False,
    ) -> None:
        if modality == self.CUSTOM:
            modality = custom_modality
        if output_type == self.CUSTOM:
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
            self.widget_unet_prediction.model_name.reset_choices()
        else:
            log(
                f"Error adding new model {new_model_name} to the list of available models: {error_msg}",
                level="error",
                thread="Add Custom Model",
            )
        self.cancel_custom_model()

    def cancel_custom_model(self, event=None):
        logger.debug("Cancel_custum_model called!")
        self.widget_add_custom_model.hide()
        self.widget_unet_prediction.show()
        # self.widget_add_custom_model_toggle.show()

    def _on_custom_modality_change(self, modality: str):
        logger.debug(f"_on_custom_modality_change called: {modality}")
        if modality == self.CUSTOM:
            self.widget_add_custom_model.custom_modality.show()
        else:
            self.widget_add_custom_model.custom_modality.hide()

    def _on_custom_output_type_change(self, output_type: str):
        logger.debug(f"_on_custom_output_type_change called: {output_type}")
        if output_type == self.CUSTOM:
            self.widget_add_custom_model.custom_output_type.show()
        else:
            self.widget_add_custom_model.custom_output_type.hide()
