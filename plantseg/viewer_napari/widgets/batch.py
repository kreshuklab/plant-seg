from pathlib import Path
from typing import Literal, Optional

import torch
from magicgui import magic_factory
from magicgui.widgets import Container, Label
from napari.components import tooltip

from plantseg import PATH_PLANTSEG_MODELS, logger
from plantseg.functionals.training.train import unet_training
from plantseg.tasks.workflow_handler import workflow_handler
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.output import Output_Tab
from plantseg.viewer_napari.widgets.utils import div
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
    def __init__(self):
        # Constants
        self.ALL_CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.MPS = ["mps"] if torch.backends.mps.is_available() else []
        self.ALL_DEVICES = self.ALL_CUDA_DEVICES + self.MPS + ["cpu"]

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

    def get_container(self):
        return Container(
            widgets=[
                div("Custom Model Training"),
                self.widget_unet_training,
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
        sparse={
            "label": "Sparse",
            "widget_type": "CheckBox",
            "value": True,
            "tooltip": "If True, use Softmax in final layer.\nElse use a Sigmoid.",
        },
        device={
            "label": "Device",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
    )
    def factory_unet_training(
        self,
        dataset: Path,
        model_name: str,
        channels,
        feature_maps,
        patch_size,
        max_num_iters: int,
        dimensionality,
        sparse,
        device,
    ) -> None:
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

        unet_training(
            dataset_dir=dataset,
            model_name=model_name,
            in_channels=channels[0],
            out_channels=channels[1],
            feature_maps=feature_maps,
            patch_size=patch_size,
            max_num_iters=max_num_iters,
            dimensionality=dimensionality,
            sparse=sparse,
            device=device,
        )
        log(f"Finished training, saved model to {checkpoint_dir}", thread="train_gui")

    def _on_dimensionality_change(self, dimensionality: Literal["3D", "2D"]):
        """Update patch size according to chosen dimensionality."""

        if dimensionality == "2D":
            ps = self.widget_unet_training.patch_size.value
            self.previous_patch_size = ps
            self.widget_unet_training.patch_size.value = [1, ps[1], ps[2]]
        else:
            self.widget_unet_training.patch_size.value = self.previous_patch_size


class Misc_Tab:
    def __init__(self, output_tab: Optional[Output_Tab] = None):
        self.batch_tab = Batch_Tab(output_tab)
        self.train_tab = Training_Tab()

    def get_container(self):
        return Container(
            widgets=[
                self.batch_tab.get_container(),
                self.train_tab.get_container(),
            ],
            labels=False,
        )
