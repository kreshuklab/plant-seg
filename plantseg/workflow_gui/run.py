from typing import Optional
from pathlib import Path
from magicgui import magicgui
import yaml
from plantseg.workflow_gui.widgets import Workflow_widgets, logger

import rich.traceback
import psygnal

rich.traceback.install(
    show_locals=True,
    suppress=["psygnal", "magicgui"],
)


class Workflow_gui(Workflow_widgets):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(config_path)

        # setup containers
        self.bottom_buttons.append(self.exit)

        # setup initial state
        if self.config is None:
            logger.debug("No config, showing loader..")
            self.show_loader()

        self.main_window.show(run=True)

    def show_loader(self):
        logger.debug("Changing to loader view")
        self.content.clear()
        self.content.append(self.loader_w)
        self.loader_w.show()

    def show_config(self):
        try:
            self.validate_config()
        except ValueError as e:
            logger.error(f"Workflow not valid, please choose a valid workflow!\n{e}")
            return

        logger.debug("Changing to config view")
        self.content.clear()
        self.fill_config_c()

    def validate_config(self):
        if not isinstance(self.config, dict):
            raise ValueError("Config was not parsed to dict")
        if "inputs" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'inputs' section."
            )
        if "infos" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'infos' section."
            )
        if "list_tasks" not in self.config:
            raise ValueError(
                "The workflow configuration does not contain an 'list_tasks' section."
            )
        if "runner" not in self.config:
            logger.warning(
                "The workflow configuration does not contain a 'runner' section. Using the default serial runner."
            )
            self.config["runner"] = "serial"


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    config = Path("examples/headless_workflow.yaml")
    Workflow_gui(config)
