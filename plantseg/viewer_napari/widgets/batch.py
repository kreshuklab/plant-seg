from pathlib import Path
from typing import Optional

from magicgui import magic_factory
from magicgui.widgets import Container, Label

from plantseg import logger
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
