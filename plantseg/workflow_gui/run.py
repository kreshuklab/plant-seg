from pathlib import Path
from typing import Optional

import psygnal
import rich.traceback
import yaml
from magicgui import magicgui
from magicgui.widgets import Container

from plantseg.workflow_gui.widgets import Workflow_widgets, logger

rich.traceback.install(
    show_locals=True,
    suppress=[psygnal],
)


class Workflow_gui(Workflow_widgets):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.config_path = config_path

        # setup initial state
        if not isinstance(self.config_path, Path):
            logger.debug("No config, showing loader..")
            self.show_loader()
        else:
            logger.debug("Config provided")
            self.loader(self.config_path)

        self.main_window.show(run=True)

    def show_loader(self):
        logger.debug("Changing to loader view")
        self.content.clear()
        self.content.append(self.loader_w)

        self.bottom_buttons.clear()
        self.bottom_buttons.append(self.exit)

        [w.show() for w in self.content]
        [w.show() for w in self.bottom_buttons]

        self.main_window.native.resize(self.main_window.native.minimumSizeHint())

    def show_config(self):
        try:
            self.validate_config()
        except ValueError as e:
            logger.error(f"Workflow not valid, please choose a valid workflow!\n{e}")
            return

        self.bottom_buttons.clear()
        self.bottom_buttons.extend((self.change_config, self.exit))

        logger.debug("Changing to config view")
        self.content.clear()
        self.fill_config_c()

        [w.show() for w in self.content]
        [w.show() for w in self.bottom_buttons]

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
    # config = Path("examples/multi.yaml")
    # config = None
    Workflow_gui(config)
