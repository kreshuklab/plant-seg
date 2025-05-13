import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import yaml
from magicgui import magic_factory, magicgui, widgets
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSlider,
    Label,
    LineEdit,
    MainWindow,
    PushButton,
)
from qtpy.QtWidgets import QLabel, QSlider, QWidget

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
        self.changing_fields = {"tasks": {}, "inputs": {}}

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
        """Fills content container with everything about one workflow."""

        logger.debug("Filling contents section")
        input_c = Container(layout="vertical", labels=False)
        self.fill_input_c(input_c)
        self.content.append(input_c)

        tasks_c = Container(layout="vertical", labels=False)
        self.fill_tasks_c(tasks_c)
        self.content.append(tasks_c)

    def fill_tasks_c(self, cont: Container, node: Optional[dict] = None, depth=0):
        """Fills tasks section of config container with all tasks trees"""

        if node is None:
            # Header, start building tree recursively
            cont.append(Label(value="Tasks:\n"))
            cont.append(Container(layout="horizontal"))
            cont = cont[-1]

            for task in self.config["list_tasks"]:
                if task["node_type"] == "root":
                    t_container = Container(
                        layout="vertical",
                        labels=True,
                        scrollable=True,
                    )
                    t_container.margins = (0, 0, 0, 0)
                    logger.debug(f"Building task tree for root {task['images_inputs']}")
                    logger.debug(t_container.margins)
                    self.fill_tasks_c(node=task, cont=t_container)
                    cont.append(t_container)

                    print(t_container.native.children())
                    children = t_container.native.children()
                    colors = [255 for _ in range(len(children))]
                    i = [0 for _ in range(len(children))]
                    while children:
                        ch = children.pop(0)
                        c = colors.pop(0)
                        j = i.pop(0)
                        if getattr(ch, "setStyleSheet", False):
                            if len(ch.children()) < 2:
                                continue
                            print(" " * j, ch, len(ch.children()))
                            ch.setStyleSheet(f"background-color: rgb({c}, {c}, {c})")
                            new_chs = ch.children()
                            colors.extend([c - 20 for _ in range(len(new_chs))])
                            i.extend([0 for _ in range(len(new_chs))])
                            children.extend(new_chs)

        # @@@ Different control widgets @@@
        else:
            label = " ".join(f"{node['func']}".split("_")[:-1])

            if node["func"] == "import_image_task":
                w = ComboBox(
                    label=label,
                    value=node["images_inputs"]["input_path"],
                    choices=list(
                        filter(
                            lambda s: s.startswith("input"),
                            self.config["inputs"][0],
                        )
                    ),
                )
                # cont.append(Label(value=label))
                cont.append(w)
                self.changing_fields["tasks"][node["id"]] = lambda: {
                    "images_inputs": {"input_path": w.value}
                }

            elif node["func"] == "gaussian_smoothing_task":
                w = FloatSlider(
                    label=label,
                    value=node["parameters"]["sigma"],
                    min=0.1,
                    max=10,
                )
                # cont.append(Label(value=label))
                cont.append(w)
                self.changing_fields["tasks"][node["id"]] = lambda: {
                    "parameters": {"sigma": w.value}
                }

            elif node["func"] == "export_image_task":
                w = ComboBox(
                    label=label,
                    value=node["images_inputs"]["export_directory"],
                    choices=list(
                        filter(
                            lambda s: s.startswith("export"),
                            self.config["inputs"][0],
                        )
                    ),
                )
                cont.append(w)
                self.changing_fields["tasks"][node["id"]] = lambda: {
                    "images_inputs": {"export_directory": w.value}
                }

            elif node["func"] == "unet_prediction_task":
                # model_name
                # device
                unet_cont = Container(label=label)
                unet_cont.append(
                    LineEdit(
                        label="Model:",
                        value=node["parameters"]["model_name"],
                    )
                )
                self.changing_fields["tasks"][node["id"]] = lambda: {
                    "parameters": {"model_name": unet_cont[-1].value}
                }
                unet_cont.append(
                    LineEdit(
                        label="Device:",
                        value=node["parameters"]["device"],
                    )
                )
                self.changing_fields["tasks"][node["id"]] = lambda: {
                    "parameters": {"device": unet_cont[-1].value}
                }
                cont.append(unet_cont)

            else:
                # catch-all
                m = (
                    " ".join(f"{node['func']}".split("_")[:-1])
                    + f"\n{node.get('parameters')}"
                )
                cont.append(Label(label="smth", value=m))

            # append next task:
            if len(node["outputs"]) == 0:
                return
            else:
                # If multiple outputs make multiple subcontainers.
                # Otherwise use parent container.
                task_containers = {}
                if len(node["outputs"]) > 1:
                    for out in node["outputs"]:
                        task_containers[out] = Container(
                            layout="vertical",
                        )
                        task_containers[out].margins = (0, 0, 0, 0)
                    cont.append(
                        Container(
                            widgets=task_containers.values(),
                            layout="horizontal",
                            labels=False,
                        )
                    )
                    cont[-1].margins = (0, 0, 0, 0)
                else:
                    task_containers[node["outputs"][0]] = cont

                # c = 255 - (depth * 20)
                # cont[-1].native.setStyleSheet(f"background-color: rgb({c}, {c}, {c})")

            for task in self.config["list_tasks"]:
                for out in node["outputs"]:
                    if out in task["images_inputs"].values():
                        self.fill_tasks_c(
                            node=task, cont=task_containers[out], depth=depth + 1
                        )

    def fill_input_c(self, cont: Container):
        """Fills the input section of the config container"""

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
        self.save.path.value = self.config_path

        save_b.changed.connect(self.save.show)

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

    @magicgui(
        main_window=True,
        call_button="Save",
        path={
            "label": "Save to",
            "mode": "w",
            "tooltip": "Choose where to save the workflow",
            "filter": "*.y[a]ml",
        },
    )
    def save(self, path: Path):
        logger.debug(f"Called save {path}")
        if path.suffix not in [".yaml", ".yml"]:
            logging.warning("Please save as a yaml file!")
            return

        output = self.config.copy()
        for field, w in self.changing_fields["inputs"].items():
            output["inputs"][0][field] = w()

        logger.debug(f"Output: {output['inputs'][0]}")

        for id, w in self.changing_fields["tasks"].items():
            for i, task in enumerate(output["list_tasks"]):
                if task["id"] == id:
                    update = w()
                    output["list_tasks"][i].update(update)

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
