from pathlib import Path
from typing import Literal, Optional

import torch
from magicgui import magic_factory
from magicgui.types import Undefined
from magicgui.widgets import Container, Label, ProgressBar
from napari.layers import Image, Labels

from plantseg import PATH_PLANTSEG_MODELS, logger
from plantseg.core.image import SemanticType
from plantseg.core.zoo import model_zoo
from plantseg.functionals.training.train import find_h5_files
from plantseg.io.h5 import read_h5_voxel_size
from plantseg.tasks.training_tasks import unet_training_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.prediction import Prediction_Widgets
from plantseg.viewer_napari.widgets.utils import div, get_layers, schedule_task


class Training_Tab:
    def __init__(self, prediction_tab: Optional[Prediction_Widgets]):
        self.prediction_tab = prediction_tab

        # Constants
        self.ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.MPS = ["mps"] if torch.backends.mps.is_available() else []
        self.ALL_DEVICES = self.ALL_CUDA_DEVICES + self.MPS + ["cpu"]
        self.CUSTOM = "Custom"

        self.previous_patch_size = [16, 64, 64]

        # initialize widgets
        self.widget_unet_training = self.factory_unet_training()
        self.widget_unet_training.self.bind(self)

        self.widget_unet_training.from_disk.changed.connect(self._on_from_disk_change)

        self.widget_unet_training.device._default_choices = self.ALL_DEVICES
        self.widget_unet_training.device.reset_choices()
        self.widget_unet_training.device.value = self.ALL_DEVICES[0]

        self.widget_unet_training.pretrained._default_choices = (
            lambda _: model_zoo.list_models()
        )
        self.widget_unet_training.pretrained.changed.connect(
            self._on_pretrained_changed
        )
        self.widget_unet_training.pretrained.reset_choices()

        self.widget_unet_training.dimensionality.changed.connect(
            self._on_dimensionality_change
        )

        self.widget_unet_training.modality.changed.connect(
            self._on_custom_modality_change
        )
        self.widget_unet_training.output_type.changed.connect(
            self._on_custom_output_type_change
        )
        self.widget_unet_training.modality._default_choices = (
            model_zoo.get_unique_modalities() + [self.CUSTOM]
        )
        self.widget_unet_training.modality.reset_choices()
        self.widget_unet_training.output_type._default_choices = (
            model_zoo.get_unique_output_types() + [self.CUSTOM]
        )
        self.widget_unet_training.output_type.reset_choices()
        self.widget_unet_training.dataset.changed.connect(self._on_dataset_change)

        self.widget_info = Label(value=f"Model dir: {PATH_PLANTSEG_MODELS}")

    def get_container(self):
        return Container(
            widgets=[
                div("Custom Model Training"),
                self.widget_unet_training,
                self.widget_info,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Start Training",
        from_disk={
            "label": "Train from",
            "value": "Disk",
            "widget_type": "RadioButtons",
            "choices": ["Disk", "GUI"],
            "orientation": "horizontal",
            "tooltip": "Whether to train from a dataset on disk or from loaded layers in Napari.",
        },
        dataset={
            "label": "Dataset",
            "value": Path.home(),
            "mode": "d",
            "tooltip": "Dataset directory. It must contain a `train` and a `val`\n"
            "directory, each containing h5 files.\n"
            "Input/Output keys must be `raw` and `label` respectively.",
        },
        image={
            "label": "Training input",
            "tooltip": "e.g. raw microscopy image",
            "visible": False,
        },
        segmentation={
            "label": "Training segmentation",
            "tooltip": "The to the training input corresponding segmentation.",
            "visible": False,
        },
        pretrained={
            "label": "Pretrained model",
            "tooltip": "Optionally select an existing model to retrain.\n"
            "Hover over the name to show the model description.\n"
            "Leave empty to create a new model.",
            "choices": [None],
            "value": None,
        },
        model_name={
            "label": "Model name",
            "value": "",
            "tooltip": "How your new model should be called.\n"
            "Can't be the name of an existing model.",
        },
        description={
            "label": "Description",
            "value": "A model trained by the user.",
            "tooltip": "Model description will be saved alongside the model.",
        },
        channels={
            "label": "In and Out Channels",
            "value": (1, 1),
            "tooltip": "Number of input and output channels",
            "widget_type": "TupleEdit",
        },
        feature_maps={
            "label": "Feature dimensions",
            "widget_type": "ListEdit",
            "value": [16],
            "tooltip": "Number of feature maps at each level of the encoder.\n"
            "If it's one number, the number of feature maps is given by\n"
            "the geometric progression: f_maps ^ k, k=1,2,3,4\n"
            "Can't be modified for pretrained models.",
        },
        patch_size={
            "label": "Patch size",
            "value": [16, 64, 64],
            "widget_type": "TupleEdit",
            "tooltip": "",
        },
        resolution={
            "label": "Data Resolution",
            "value": [1.0, 1.0, 1.0],
            "widget_type": "TupleEdit",
            "tooltip": "Voxel size in um of the training data.\n"
            "Is initialized correctly from the chosen data if possible.",
        },
        max_num_iters={
            "label": "Max iterations",
            "value": 100,
            "max": 100000000,
            "min": 1,
            "tooltip": "Maximum number of iterations after which the training\n"
            "will be stopped. Stops earlier if the accuracy converges.",
        },
        dimensionality={
            "label": "Dimensionality",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "value": "3D",
            "choices": ["3D", "2D"],
            "tooltip": "Train a 3D unet or a 2D unet",
        },
        modality={
            "label": "Microscopy modality",
            "tooltip": "Modality of the model (e.g. confocal, light-sheet ...).",
            "widget_type": "ComboBox",
            "value": None,
        },
        custom_modality={
            "label": "Custom modality",
            "value": Undefined,
            "visible": False,
        },
        output_type={
            "label": "Prediction type",
            "widget_type": "ComboBox",
            "tooltip": "Type of prediction (e.g. cell boundaries prediction or nuclei...).",
            "value": None,
        },
        custom_output_type={
            "label": "Custom type",
            "value": Undefined,
            "visible": False,
        },
        sparse={
            "label": "Sparse",
            "widget_type": "RadioButtons",
            "choices": [False, True],
            "value": False,
            "orientation": "horizontal",
            "tooltip": "If True, use Softmax in final layer.\nElse use a Sigmoid.",
        },
        device={
            "label": "Device",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        pbar={"label": "Training in progress", "max": 0, "min": 0, "visible": False},
    )
    def factory_unet_training(
        self,
        from_disk: str,
        dataset: Optional[Path],
        image: Optional[Image],
        segmentation: Optional[Labels],
        pretrained: Optional[str],
        model_name: str,
        description: str,
        channels,
        feature_maps,
        patch_size,
        resolution,
        max_num_iters: int,
        dimensionality,
        sparse,
        device,
        modality: Optional[str] = None,
        custom_modality: str = "",
        output_type: Optional[str] = None,
        custom_output_type: str = "",
        pbar: Optional[ProgressBar] = None,
    ) -> None:
        """Train a boundary prediction unet"""
        if from_disk == "Disk":
            image = None
            segmentation = None
            if dataset is None:
                log("Please choose a dataset to load!", thread="train_gui")
                return

        else:
            dataset = None
            if image is None or segmentation is None:
                log(
                    "Please choose a raw image and a segmentation to train!",
                    thread="train_gui",
                )
                return

        if modality is None or output_type is None:
            log("Choose a modality and a prediction type!", thread="train_gui")
            return
        if modality == self.CUSTOM:
            modality = custom_modality
            if len(modality) == 0:
                log("Custom modality can't be empty!", thread="train_gui")
                return
        if output_type == self.CUSTOM:
            output_type = custom_output_type
            if len(output_type) == 0:
                log("Custom prediction type can't be empty!", thread="train_gui")
                return
        if len(model_name) == 0:
            log("Please choose a model name!", thread="train_gui")
            return

        # Enable geometric progression by setting type to int
        if len(feature_maps) == 1:
            feature_maps = feature_maps[0]

        checkpoint_dir = PATH_PLANTSEG_MODELS / model_name
        if checkpoint_dir.exists():
            log(
                "Model exists already! Please choose another name.",
                thread="train_gui",
                level="WARNING",
            )
            return

        pre_model_path = None
        if pretrained is not None:
            model, model_config, pre_model_path = model_zoo.get_model_by_name(
                pretrained
            )

        widgets_to_reset = [
            self.widget_unet_training.pretrained,
        ]
        if self.prediction_tab:
            self.prediction_tab.widget_unet_prediction.model_name

        log("Starting training task", thread="train_gui")
        schedule_task(
            task=unet_training_task,
            task_kwargs={
                "dataset_dir": dataset,
                "image": image,
                "segmentation": segmentation,
                "model_name": model_name,
                "in_channels": channels[0],
                "out_channels": channels[1],
                "feature_maps": feature_maps,
                "patch_size": patch_size,
                "max_num_iters": max_num_iters,
                "dimensionality": dimensionality,
                "sparse": sparse,
                "device": device,
                "modality": modality,
                "output_type": output_type,
                "description": description,
                "resolution": resolution,
                "pre_trained": pre_model_path,
                "widgets_to_reset": widgets_to_reset,
                "_pbar": pbar,
                "_to_hide": [self.widget_unet_training.call_button],
            },
        )

    def _on_from_disk_change(self, from_disk: str):
        logger.debug(f"_on_from_disk_change called: {from_disk}")
        if from_disk == "Disk":
            self.widget_unet_training.image.hide()
            self.widget_unet_training.segmentation.hide()
            self.widget_unet_training.dataset.show()
        else:
            self.widget_unet_training.dataset.hide()
            self.widget_unet_training.image.show()
            self.widget_unet_training.segmentation.show()

    def update_layer_selection(self, event):
        """Updates layer drop-down menus"""
        logger.debug(
            f"Updating segmentation layer selection: {event.value}, {event.type}"
        )
        raws = get_layers(SemanticType.RAW)
        segmentations = get_layers(SemanticType.SEGMENTATION)

        self.widget_unet_training.image.choices = raws
        self.widget_unet_training.segmentation.choices = segmentations

        if raws and segmentations:
            self.widget_unet_training.from_disk.value = "GUI"

    def _on_dimensionality_change(self, dimensionality: Literal["3D", "2D"]):
        """Update patch size according to chosen dimensionality."""

        if dimensionality == "2D":
            ps = self.widget_unet_training.patch_size.value
            self.previous_patch_size = ps
            self.widget_unet_training.patch_size.value = [1, ps[1], ps[2]]
        else:
            self.widget_unet_training.patch_size.value = self.previous_patch_size

    def _on_custom_modality_change(self, modality: str):
        logger.debug(f"_on_custom_modality_change called: {modality}")
        if modality == self.CUSTOM:
            self.widget_unet_training.custom_modality.show()
        else:
            self.widget_unet_training.custom_modality.hide()

    def _on_custom_output_type_change(self, output_type: str):
        logger.debug(f"_on_custom_output_type_change called: {output_type}")
        if output_type == self.CUSTOM:
            self.widget_unet_training.custom_output_type.show()
        else:
            self.widget_unet_training.custom_output_type.hide()

    def _on_dataset_change(self, dataset_dir: Path):
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return
        if not all((dataset_dir / d).exists() for d in ["train", "val"]):
            return

        logger.debug(f"_on_dataset_change called: {dataset_dir}")

        h5s = find_h5_files(dataset_dir / "train")
        if len(h5s) < 1:
            logger.debug("_on_dataset_change: no h5 files found")
            return

        voxel_size = read_h5_voxel_size(h5s[0], "raw").voxels_size
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)

        self.widget_unet_training.resolution.value = voxel_size
        logger.debug(f"Resolution of training data: {voxel_size}")

    def _on_pretrained_changed(self, model_name: str | None):
        logger.debug(f"_on_model_name_changed called: {model_name}")

        if model_name is None:
            self.description = "No description available for this model."
            self.widget_unet_training.feature_maps.enabled = True
        else:
            self.widget_unet_training.feature_maps.enabled = False
            model, model_config, pre_model_path = model_zoo.get_model_by_name(
                model_name
            )
            logger.info(f"Selected model config: {model_config}")
            if isinstance(model_config["f_maps"], list):
                self.widget_unet_training.feature_maps.value = model_config["f_maps"]
            else:
                self.widget_unet_training.feature_maps.value = [model_config["f_maps"]]

            self.description = model_zoo.get_model_description(model_name)

        self.widget_unet_training.pretrained.tooltip = (
            "Select an existing model to retrain. Current model description:"
            f"\n\n{self.description}"
        )
