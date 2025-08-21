from pathlib import Path
from typing import Literal, Optional

import torch
from magicgui import magic_factory
from magicgui.types import Undefined
from magicgui.widgets import Container, Label, ProgressBar
from napari.components import tooltip

from plantseg import PATH_PLANTSEG_MODELS, logger
from plantseg.core.zoo import model_zoo
from plantseg.functionals.training.train import find_h5_files, unet_training
from plantseg.io.h5 import read_h5_voxel_size
from plantseg.tasks.training_tasks import unet_training_task
from plantseg.tasks.workflow_handler import task_tracker, workflow_handler
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.prediction import Prediction_Widgets
from plantseg.viewer_napari.widgets.utils import div, schedule_task
from plantseg.workflow_gui.editor import Workflow_gui


class Batch_Tab:
    def __init__(self, output_tab: Optional[Output_Tab] = None):
        self.widget_export_workflow = self.factory_export_headless_workflow()
        self.widget_export_workflow.self.bind(self)
        self.widget_export_workflow.hide()

        self.widget_edit_worflow = self.factory_edit_worflow()
        self.widget_edit_worflow.self.bind(self)

        self.widget_export_placeholder = Label(
            value="Export an image before saving the workflow\nfor batch execution!"
        )

        if output_tab:
            output_tab.successful_export.connect(self.toggle_export_vis)

    def get_container(self):
        return Container(
            widgets=[
                div("Export Batch Workflow"),
                self.widget_export_placeholder,
                self.widget_export_workflow,
                div("Edit Batch Workflow"),
                self.widget_edit_worflow,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Export Workflow",
        directory={
            "label": "Export directory",
            "mode": "d",
            "tooltip": "Select the directory where the workflow will be exported",
        },
        workflow_name={
            "label": "Workflow name",
            "tooltip": "Name of the exported workflow file.",
        },
    )
    def factory_export_headless_workflow(
        self,
        directory: Path = Path.home(),
        workflow_name: str = "headless_workflow.yaml",
    ) -> None:
        """Save the workflow as a yaml file"""

        if not workflow_name.endswith(".yaml"):
            workflow_name = f"{workflow_name}.yaml"

        workflow_path = directory / workflow_name
        workflow_handler.save_to_yaml(path=workflow_path)

        log(f"Workflow saved to {workflow_path}", thread="Export stacks", level="info")

    @magic_factory(
        call_button="Edit a Workflow",
    )
    def factory_edit_worflow(self) -> None:
        log("Starting workflow editor", thread="Workflow")
        Workflow_gui()

    def toggle_export_vis(self):
        logger.debug("toggle export called!")
        self.widget_export_placeholder.hide()
        self.widget_export_workflow.show()


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
        self.widget_unet_training.device._default_choices = self.ALL_DEVICES
        self.widget_unet_training.device.reset_choices()
        self.widget_unet_training.device.value = self.ALL_DEVICES[0]

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
        dataset={
            "label": "Dataset",
            "value": Path.home(),
            "mode": "d",
            "tooltip": "Dataset directory should contain a `train` and a `val`\n"
            "directory, each containing h5 files.",
        },
        model_name={
            "label": "Model name",
            "value": "",
            "tooltip": "How your new model should be called.\n"
            "Can't be the name of an existing model.",
        },
        description={"label": "Description", "value": "A model trained by the user."},
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
            "the geometric progression: f_maps ^ k, k=1,2,3,4",
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
        dataset: Path,
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

        if not all((dataset / d).exists() for d in ["train", "val"]):
            log(
                "Dataset dir must contain a train and a val directory,\n"
                "each containing h5 files!",
                thread="train_gui",
            )
            return

        log("Starting training task", thread="train_gui")
        schedule_task(
            task=unet_training_task,
            task_kwargs={
                "dataset_dir": dataset,
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
                "prediction_tab": self.prediction_tab,
                "_pbar": pbar,
                "_to_hide": [self.widget_unet_training.call_button],
            },
        )

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


class Misc_Tab:
    def __init__(
        self,
        output_tab: Optional[Output_Tab] = None,
        prediction_tab: Optional[Prediction_Widgets] = None,
    ):
        self.batch_tab = Batch_Tab(output_tab)
        self.train_tab = Training_Tab(prediction_tab)

    def get_container(self):
        return Container(
            widgets=[
                self.batch_tab.get_container(),
                self.train_tab.get_container(),
            ],
            labels=False,
        )
