from pathlib import Path
from typing import Literal, Optional

import torch
from magicgui import magic_factory
from magicgui.types import Undefined
from magicgui.widgets import Container, Label, ProgressBar
from napari.layers import Image, Labels

from plantseg import PATH_PLANTSEG_MODELS, logger
from plantseg.core.image import ImageLayout, PlantSegImage, SemanticType
from plantseg.core.zoo import model_zoo
from plantseg.functionals.training.model import UNet2D, UNet3D
from plantseg.functionals.training.train import find_h5_files
from plantseg.io.h5 import read_h5_shape, read_h5_voxel_size
from plantseg.tasks.training_tasks import unet_training_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.prediction import Prediction_Widgets
from plantseg.viewer_napari.widgets.utils import div, get_layers, schedule_task


class Training_Tab:
    def __init__(self, prediction_tab: Optional[Prediction_Widgets]):
        self.prediction_tab = prediction_tab
        self.in_shape = None
        self.out_shape = None

        # Constants
        self.ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.MPS = ["mps"] if torch.backends.mps.is_available() else []
        self.MPS = []  # MPS does lack some necessary ops #385
        self.ALL_DEVICES = self.ALL_CUDA_DEVICES + self.MPS + ["cpu"]
        self.CUSTOM = "Custom"

        self.previous_patch_size = [16, 64, 64]

        # initialize widgets
        self.widget_unet_training = self.factory_unet_training()
        self.widget_unet_training.self.bind(self)

        self.widget_unet_training.insert(0, div("Training Data", False))
        self.widget_unet_training.insert(8, div("Model", False))
        self.widget_unet_training.insert(15, div("Meta Data", False))

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
        self.widget_unet_training.image.changed.connect(self._on_image_change)
        self.widget_unet_training.segmentation.changed.connect(
            self._on_segmentation_change
        )

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
            "choices": ["Disk", "Current Data"],
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
            "enabled": False,
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
            "enabled": False,
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
            "enabled": False,
            "visible": False,  # output channels not supported, so it is always known
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
        device={
            "label": "Device",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        pbar={"label": "Training in progress", "max": 0, "min": 0, "visible": False},
    )
    def factory_unet_training(
        self,
        # data
        from_disk: str,
        dataset: Optional[Path],
        image: Optional[Image],
        segmentation: Optional[Labels],
        channels,
        resolution,
        dimensionality,
        # model
        pretrained: Optional[str],
        feature_maps,
        patch_size,
        max_num_iters: int,
        device,
        # metadata
        model_name: str,
        description: str,
        modality: Optional[str],
        custom_modality: str,
        output_type: Optional[str],
        custom_output_type: str,
        pbar: Optional[ProgressBar],
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
            widgets_to_reset.append(
                self.prediction_tab.widget_unet_prediction.model_name
            )

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
            self.widget_unet_training.from_disk.value = "Current Data"

    def _on_dimensionality_change(self, dimensionality: Literal["3D", "2D"]):
        """Update patch size according to chosen dimensionality."""
        logger.debug(f"_on_dimensionality_change called with {dimensionality}")

        # Patch size:
        if dimensionality == "2D":
            ps = self.widget_unet_training.patch_size.value
            self.previous_patch_size = ps
            self.widget_unet_training.patch_size.value = [1, ps[1], ps[2]]
        else:
            self.widget_unet_training.patch_size.value = self.previous_patch_size
        self.update_channels()

    def update_channels(self):
        """Updates the number of input and output channels

        Determination based on self.in_shape and self.out_shape, which are set
        on data change.
        This function would support output channels, but currently the training
        does not, therefore the `dimensionality` field is not visible, as it is
        only needed in the case of 3d in- and 3d output with an uncertain number
        of output channels.
        """
        try:
            logger.debug("update_channels called")
            dimensionality = self.widget_unet_training.dimensionality
            ch = self.widget_unet_training.channels
            if self.in_shape is None or self.out_shape is None:
                return
            if len(self.in_shape) == 2:
                if len(self.out_shape) == 2:
                    ch.value = (1, 1)
                elif len(self.out_shape) == 3:
                    ch.value = (1, self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                elif len(self.out_shape) == 4:
                    raise ValueError("For 2D input only 2D output is supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")

            elif len(self.in_shape) == 3:
                if len(self.out_shape) == 2:
                    # must be 2D!
                    ch.value = (self.in_shape[0], 1)

                elif len(self.out_shape) == 3:
                    if dimensionality.value == "3D":
                        ch.value = (1, 1)
                    elif dimensionality.value == "2D":
                        ch.value = (self.in_shape[0], self.out_shape[0])
                        raise ValueError("No channels in output supported!")

                elif len(self.out_shape) == 4:
                    # must be 3D!
                    ch.value = (1, self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")

            elif len(self.in_shape) == 4:
                if len(self.out_shape) == 2:
                    raise ValueError("For 3D input only 3D output is supported!")
                elif len(self.out_shape) == 3:
                    ch.value = (self.in_shape[0], 1)
                elif len(self.out_shape) == 4:
                    ch.value = (self.in_shape[0], self.out_shape[0])
                    raise ValueError("No channels in output supported!")
                else:
                    raise ValueError(f"Output shape {self.out_shape} not supported!")
            else:
                raise ValueError(f"Input shape {self.in_shape} not supported!")
            logger.debug(f"Determined channels: {ch.value}")
        except ValueError as e:
            log(f"Error: {e}", thread="training", level="ERROR")

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
        """Updates resolution, dimensionality and channels"""
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return
        if not all((dataset_dir / d).exists() for d in ["train", "val"]):
            return

        logger.debug(f"_on_dataset_change called: {dataset_dir}")

        h5s = find_h5_files(dataset_dir / "train")
        if len(h5s) < 1:
            logger.debug("_on_dataset_change: no h5 files found")
            return

        # get resolution
        voxel_size = read_h5_voxel_size(h5s[0], "raw").voxels_size
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
            self.widget_unet_training.resolution.enabled = True
        else:
            self.widget_unet_training.resolution.enabled = False

        self.widget_unet_training.resolution.value = voxel_size
        logger.debug(f"Resolution of training data: {voxel_size}")

        # get channels
        self.in_shape = read_h5_shape(h5s[0], key="raw")
        self.out_shape = read_h5_shape(h5s[0], key="label")
        logger.debug(
            f"In/Out shape determined from dataset: {self.in_shape}, {self.out_shape}"
        )
        self.update_dimensionality()

    def update_dimensionality(self):
        """Updates and enables/disables the dimensionality input

        This should be called whenever the input is changed, after
        setting self.in_shape and self.out_shape.

        Triggers _on_dimensionality_change, which calls update_channels.
        """
        logger.debug(
            f"update_dimensionality called, in/out: {self.in_shape}, {self.out_shape}"
        )
        dimensionality = self.widget_unet_training.dimensionality

        if self.in_shape is None or self.out_shape is None:
            return
        if len(self.in_shape) == 2:
            dimensionality.enabled = False
            dimensionality.value = "2D"
            if len(self.out_shape) > 3:
                raise ValueError(
                    f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
                )

        elif len(self.in_shape) == 3:
            if len(self.out_shape) == 2:
                dimensionality.enabled = False
                dimensionality.value = "2D"
            elif len(self.out_shape) == 3:
                # Dimensionality only ambiguouse if output could have channels
                # Which it can't atm -> dimensionality not visible
                dimensionality.enabled = True
                dimensionality.value = "3D"
            elif len(self.out_shape) == 4:
                dimensionality.enabled = False
                dimensionality.value = "3D"

        elif len(self.in_shape) == 4:
            dimensionality.enabled = False
            dimensionality.value = "3D"

            if len(self.out_shape) < 3:
                raise ValueError(
                    f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
                )

        if (not 1 < len(self.in_shape) < 5) or (not 1 < len(self.out_shape) < 5):
            raise ValueError(
                f"Input/Output shapes not supported: {self.in_shape}, {self.out_shape}"
            )
        # update_channels called from setting dimensionality
        dimensionality.changed.emit(dimensionality.value)

    def _on_image_change(self, image: Image):
        """Update resolution, dimensionality and input channels on image change"""
        pl_image = PlantSegImage.from_napari_layer(image)
        self.widget_unet_training.resolution.value = pl_image.voxel_size
        ch_dim = pl_image.channel_axis
        if ch_dim and pl_image.image_layout == ImageLayout.ZCYX:
            raise ValueError("Only CZYX image layout supported for 4D training.")
        self.in_shape = pl_image.shape
        self.update_dimensionality()

    def _on_segmentation_change(self, seg: Labels):
        """Update resolution, dimensionality and output channels on image change"""
        pl_image = PlantSegImage.from_napari_layer(seg)
        ch_dim = pl_image.channel_axis
        if ch_dim:
            raise ValueError("No channels in output supported.")
        self.out_shape = pl_image.shape
        self.update_dimensionality()

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

            model_channels = (model_config["in_channels"], model_config["out_channels"])
            if model_channels != self.widget_unet_training.channels.value:
                log(
                    "Model incompatible chosen data!\nModel channels: "
                    f"{model_channels}\nData channels: "
                    f"{self.widget_unet_training.channels.value}",
                    thread="training",
                    level="ERROR",
                )
            if (
                isinstance(model, UNet3D)
                and self.widget_unet_training.dimensionality.value != "3D"
            ):
                log(
                    "Model incompatible with chosen data!\n"
                    "Model is a 3D model, but data is "
                    f"{self.widget_unet_training.dimensionality.value}",
                    thread="training",
                    level="ERROR",
                )
            elif (
                isinstance(model, UNet2D)
                and self.widget_unet_training.dimensionality.value != "2D"
            ):
                log(
                    "Model incompatible with chosen data!\n"
                    "Model is a 2D model, but data is "
                    f"{self.widget_unet_training.dimensionality.value}",
                    thread="training",
                    level="ERROR",
                )

        self.widget_unet_training.pretrained.tooltip = (
            "Select an existing model to retrain. Current model description:"
            f"\n\n{self.description}"
        )
