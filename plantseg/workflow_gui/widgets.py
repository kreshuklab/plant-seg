from pathlib import Path
from typing import Optional
from magicgui import magicgui
from magicgui.widgets import Container, MainWindow, EmptyWidget, Label
import yaml
import logging

logger = logging.getLogger(__name__)


class Workflow_widgets:
    def __init__(self, config: Optional[Path]):
        self.main_window = MainWindow(layout="vertical", labels=False)
        # self.main_window.create_menu_item(
        #     menu_name="File", item_name="Exit", callback=self.exit
        # )
        self.content = Container(layout="horizontal", labels=False)
        self.bottom_buttons = Container(layout="horizontal", labels=False)
        self.main_window.extend([self.content, self.bottom_buttons])
        self._config = EmptyWidget()
        if isinstance(config, Path):
            print("calling with: ", config)
            self.loader(config)

        self.exit.show()

    @property
    def config(self):
        return self._config.value

    @config.setter
    def config(self, value):
        self._config.set_value(value)

    @magicgui(call_button="Exit")
    def exit(self):
        raise SystemExit

    @magicgui(
        call_button="Load",
        config_path={
            "label": "Path",
            "mode": "r",
            "tooltip": "Choose yaml to load",
            "filter": "*.y[a]ml",
        },
    )
    def loader_w(self, config_path: Path):
        self.loader(config_path)

    def loader(self, config_path: Path):
        logger.info(f"Loading {config_path}")
        if not config_path.is_file():
            logger.error("Please provide a yaml file!")
            return
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.show_config()

    def fill_config_c(self):
        logger.debug("Filling contents section")
        input_c = Container(layout="vertical")
        self.fill_input_c(input_c)
        self.content.append(input_c)
        [w.show() for w in self.content]

    def fill_input_c(self, cont: Container):
        logger.debug("Filling input section")
        inputs = self.config["inputs"][0]
        for input in inputs:
            logger.debug(f"{type(input)} {input}")
            cont.append(Label(value=f"input: {input}"))
            pass
        [w.show() for w in cont]

    def tasks_w(self, cont: Container):
        logger.debug("Filling tasks section")
        pass
