import logging
from abc import abstractmethod
from pathlib import Path

import yaml
from magicgui import magic_factory, magicgui, widgets
from magicgui.widgets import Container, Label, MainWindow, PushButton
from numpy import ma

logger = logging.getLogger(__name__)


class Workflow_widgets:
    def __init__(self):
        self.main_window = MainWindow(layout="vertical", labels=False)
        # self.main_window.create_menu_item(
        #     menu_name="File", item_name="Exit", callback=self.exit
        # )
        self.content = Container(layout="horizontal", labels=False)
        self.bottom_buttons = Container(layout="horizontal", labels=False)
        self.main_window.extend([self.content, self.bottom_buttons])
        self.changing_fields = {}

    @magicgui(call_button="Exit")
    def exit(self):
        raise SystemExit

    @magicgui(call_button="Load new config")
    def change_config(self):
        self.show_loader()

    @magicgui(
        call_button="Load",
        config_path={
            "widget_type": "FileEdit",
            "label": "Path",
            "mode": "r",
            "tooltip": "Choose yaml to load",
            "filter": "*.y[a]ml",
        },
    )
    def loader_w(self, config_path: Path):
        """Widget wrapper around loader"""
        self.loader(config_path)

    def loader(self, config_path: Path):
        """Loads a workflow from a yaml file"""
        logger.info(f"Loading {config_path}")

        if not (config_path.exists() and config_path.suffix in [".yaml", ".yml"]):
            logger.error("Please provide a yaml file!")
            self.show_loader()
            return
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.show_config()

    def fill_config_c(self):
        logger.debug("Filling contents section")
        input_c = Container(layout="vertical", labels=False)
        self.fill_input_c(input_c)
        self.content.append(input_c)
        # [w.show() for w in self.content]

    def fill_input_c(self, cont: Container):
        logger.debug("Filling input section")

        # TODO:The inputs section is currently a list of dicts, should be just a dict (#429)
        inputs = self.config["inputs"][0]
        cont.append(Label(value="I/O:\n"))
        field_tracker = self.changing_fields["inputs"] = {}

        for io_type, path in inputs.items():
            if io_type.startswith("name_pattern"):
                w = self.io_name(
                    name={
                        "tooltip": self.config["infos"]["inputs_schema"][io_type][
                            "description"
                        ],
                        "widget_type": "LineEdit",
                        "label": io_type,
                        "value": path,
                    }
                )
                cont.append(w)
                field_tracker[io_type] = w
                w.self.bind(self)
                continue

            file_mode = "r"
            if io_type.startswith("export_directory"):
                file_mode = "d"

            w = self.io_item(
                path={
                    "tooltip": self.config["infos"]["inputs_schema"][io_type][
                        "description"
                    ],
                    "widget_type": "FileEdit",
                    "label": io_type,
                    "value": path,
                    "mode": file_mode,
                }
            )
            cont.append(w)
            field_tracker[io_type] = w
            w.self.bind(self)

        reset_b = PushButton(text="Reset to file")
        reset_b.changed.connect(self.show_config)

        save_b = PushButton(text="Save to..")
        save = self.save(
            path={
                "value": self.config_path,
                "label": "Save to",
                "mode": "w",
                "tooltip": "Choose where to save the workflow",
                "filter": "*.y[a]ml",
            },
        )

        save_b.changed.connect(save.show)
        save.self.bind(self)

        controls_c = Container(
            widgets=[reset_b, save_b],
            layout="horizontal",
        )

        [w.show() for w in controls_c]
        cont.append(controls_c)

    @magic_factory(call_button=False)
    def io_item(self, path: Path):
        logger.debug(f"Called io_item {path}")
        return str(path)

    @magic_factory(call_button=False)
    def io_name(self, name: str):
        logger.debug(f"Called io_name {name}")
        return name

    @magic_factory(main_window=True)
    def save(self, path: Path):
        logger.debug(f"Called save {path}")
        if path.suffix not in [".yaml", ".yml"]:
            logging.warning("Please save as a yaml file!")
            return

        output = self.config.copy()
        for field, w in self.changing_fields["inputs"].items():
            output["inputs"][0][field] = w()

        logger.debug(f"Output: {output['inputs'][0]}")

        for id, w in self.changing_fields["tasks"]:
            pass  # TODO: Implement

        with open(path, "w") as f:
            f.write(yaml.safe_dump(output))
            logger.debug(f"Successfully written to {path}")

        save.hide()  # type: ignore # noqa

    def tasks_w(self, cont: Container):
        logger.debug("Filling tasks section")
        pass

    @abstractmethod
    def show_config(self):
        pass

    @abstractmethod
    def show_loader(self):
        pass
