import webbrowser
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from magicgui import magic_factory
from magicgui.widgets import Container, Label, PushButton
from napari.layers import Image, Labels, Layer

from plantseg import logger
from plantseg.core.image import ImageLayout, ImageType, PlantSegImage, SemanticType
from plantseg.io import H5_EXTENSIONS, ZARR_EXTENSIONS
from plantseg.io.h5 import list_h5_keys
from plantseg.io.zarr import list_zarr_keys
from plantseg.tasks.dataprocessing_tasks import set_voxel_size_task
from plantseg.tasks.io_tasks import export_image_task, import_image_task
from plantseg.viewer_napari.widgets.utils import _return_value_if_widget, schedule_task


class PathMode(Enum):
    FILE = "tiff, h5, etc.."
    DIR = "zarr"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


class Input_Tab:
    def __init__(self):
        self.current_dataset_keys: Sequence[Optional[str]] = [None]

        # @@@@@ Open File @@@@@
        self.widget_open_file = self.factory_open_file()
        self.widget_open_file.self.bind(self)

        # self.widget_open_file.path_mode.changed.connect(self._on_path_mode_changed)
        self.widget_open_file.path.changed.connect(self._on_path_changed)
        self.widget_open_file.button_key_refresh.changed.connect(
            self._on_refresh_keys_button
        )

        self.widget_open_file.dataset_key.choices = self.current_dataset_keys
        self.widget_open_file.dataset_key.changed.connect(self._on_dataset_key_changed)
        self.widget_open_file.called.connect(self._on_done)

        # @@@@@@ Set Voxel size @@@@@
        self.widget_set_voxel_size = self.factory_set_voxel_size()
        self.widget_set_voxel_size.self.bind(self)
        self.widget_set_voxel_size.hide()

        # self.widget_set_voxel_size.called.connect(self._on_set_voxel_size_layer_done)

        # @@@@@ Show info @@@@@
        self.widget_info = Label(
            value="Select layer to show information here...",
            label="Infos",
            tooltip="Information about the selected layer.",
        )
        self.widget_info.hide()

        # @@@@@ Show details @@@@@
        self.widget_details_layer_select = self.factory_details_layer_select()
        self.widget_details_layer_select.self.bind(self)
        self.widget_details_layer_select.layer.changed.connect(
            self._on_details_layer_select_changed
        )

        self.docs = Docs_Container()

        # self.widget_show_info.layer.changed.connect(self._on_info_layer_changed)

    def get_container(self):
        return Container(
            widgets=[
                self.docs.get_doc_container(),
                self.widget_open_file,
                self.widget_details_layer_select,
                self.widget_info,
                self.widget_set_voxel_size,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Open File",
        path={
            "value": Path.home(),
            "label": "File path\n(tiff, h5, zarr, png, jpg)",
            "mode": "r",
            "tooltip": "Select a file to be imported, the file can be a tiff, h5, png, jpg.",
        },
        new_layer_name={
            "value": "",
            "label": "Layer name",
            "tooltip": "Define the name of the output layer, default is either image or label.",
        },
        layer_type={
            "value": ImageType.IMAGE.value,
            "label": "Layer type",
            "tooltip": "Select if the image is a normal image or a segmentation",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": ImageType.to_choices(),
        },
        dataset_key={
            "label": "Key (h5/zarr only)",
            "widget_type": "ComboBox",
            "choices": [],  # set self.get_current_dataset_keys
            "tooltip": "Key to be loaded from h5",
            "visible": False,
        },
        button_key_refresh={
            "label": "Refresh keys",
            "widget_type": "PushButton",
            "visible": False,
        },
        stack_layout={
            "value": ImageLayout.ZYX.value,
            "label": "Stack layout",
            "choices": ImageLayout.to_choices(),
            "tooltip": "Stack layout",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        update_other_widgets={
            "value": True,
            "visible": False,
            "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
        },
    )
    def factory_open_file(
        self,
        path: Path,
        dataset_key: str,
        button_key_refresh: bool,
        stack_layout: str,
        layer_type: str,
        new_layer_name: str,
        update_other_widgets: bool,
    ) -> None:
        """Open a file and return a napari layer."""

        if layer_type == ImageType.IMAGE.value:
            semantic_type = SemanticType.RAW
        elif layer_type == ImageType.LABEL.value:
            semantic_type = SemanticType.SEGMENTATION
        else:
            raise ValueError(f"Unknown layer type {layer_type}")

        widgets_to_update = [
            self.widget_details_layer_select.layer,
            # widget_unet_prediction.image,
        ]

        return schedule_task(
            import_image_task,
            task_kwargs={
                "input_path": path,
                "key": dataset_key,
                "image_name": new_layer_name,
                "semantic_type": semantic_type,
                "stack_layout": stack_layout,
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )

    def generate_layer_name(self, path: Path, dataset_key: str) -> str:
        dataset_key = dataset_key.replace("/", "_")
        return path.stem + dataset_key

    def look_up_dataset_keys(self, path: Path):
        path = _return_value_if_widget(path)
        ext = path.suffix.lower()

        if ext in H5_EXTENSIONS:
            self.widget_open_file.dataset_key.show()
            self.widget_open_file.button_key_refresh.show()
            dataset_keys = list_h5_keys(path)

        elif ext in ZARR_EXTENSIONS:
            self.widget_open_file.dataset_key.show()
            self.widget_open_file.button_key_refresh.show()
            dataset_keys = list_zarr_keys(path)

        else:
            self.widget_open_file.new_layer_name.value = self.generate_layer_name(
                path, ""
            )
            self.widget_open_file.dataset_key.hide()
            self.widget_open_file.button_key_refresh.hide()
            return

        self.current_dataset_keys = dataset_keys.copy()  # Update the global variable
        self.widget_open_file.dataset_key.choices = dataset_keys
        if dataset_keys == [None]:
            self.widget_open_file.dataset_key.hide()
            self.widget_open_file.button_key_refresh.hide()
        if self.widget_open_file.dataset_key.value not in dataset_keys:
            self.widget_open_file.dataset_key.value = dataset_keys[0]
            self.widget_open_file.new_layer_name.value = self.generate_layer_name(
                path, self.widget_open_file.dataset_key.value
            )

    def _on_path_mode_changed(self, path_mode: str):
        logger.debug("_on_path_mode_changed called!")
        path_mode = _return_value_if_widget(path_mode)
        if path_mode == PathMode.FILE.value:  # file
            self.widget_open_file.path.mode = "r"
            self.widget_open_file.path.label = "File path\n(.tiff, .h5, .png, .jpg)"
        elif path_mode == PathMode.DIR.value:  # directory case
            self.widget_open_file.path.mode = "d"
            self.widget_open_file.path.label = "Zarr path\n(.zarr)"

    def _on_path_changed(self, path: Path):
        logger.debug("_on_path_changed called!")
        self.look_up_dataset_keys(path)

    def _on_refresh_keys_button(self, press: bool):
        logger.debug("_on_refresh_keys_button called!")
        self.look_up_dataset_keys(self.widget_open_file.path.value)

    def _on_dataset_key_changed(self, dataset_key: str):
        logger.debug("_on_dataset_key_changed called!")
        dataset_key = _return_value_if_widget(dataset_key)
        if dataset_key:
            self.widget_open_file.new_layer_name.value = self.generate_layer_name(
                self.widget_open_file.path.value, dataset_key
            )

    def _on_done(self):
        logger.debug("_on_done called!")
        self.look_up_dataset_keys(self.widget_open_file.path.value)

    @magic_factory(
        call_button="Set Voxel Size",
        voxel_size={
            "label": "Voxel size [um]",
            "tooltip": "Set the voxel size in micrometers.",
        },
    )
    def factory_set_voxel_size(
        self,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Set the voxel size of the selected layer."""
        layer = self.widget_details_layer_select.layer.value
        if layer is None:
            raise ValueError("No layer selected.")

        assert isinstance(layer, (Image, Labels)), (
            "Only Image and Labels layers are supported for PlantSeg voxel size."
            f"layer was {layer}, type: {type(layer)}"
        )
        ps_image = PlantSegImage.from_napari_layer(layer)
        return schedule_task(
            set_voxel_size_task,
            task_kwargs={
                "image": ps_image,
                "voxel_size": voxel_size,
            },
            widgets_to_update=[self.widget_details_layer_select.layer],
        )

    def _on_details_layer_select_changed(self, layer: Optional[Layer]):
        logger.debug(f"_on_details_layer_select_changed called for layer {layer}!")

        if layer is None:
            self.widget_set_voxel_size.hide()
            self.widget_info.hide()
            return

        if not (isinstance(layer, Labels) or isinstance(layer, Image)):
            logger.debug(f"Can't show info for {layer}")
            raise ValueError("Info can only be shown for Image or Labels layers.")

        self.widget_details_layer_select.show()
        self.widget_set_voxel_size.show()
        self.widget_info.show()

        ps_image = PlantSegImage.from_napari_layer(layer)
        if ps_image.has_valid_voxel_size():
            voxel_size_formatted = "("
            for vs in ps_image.voxel_size:
                voxel_size_formatted += f"{vs:.2f}, "

            voxel_size_formatted = (
                voxel_size_formatted[:-2] + f") {ps_image.voxel_size.unit}"
            )
        else:
            voxel_size_formatted = "None"

        str_info = (
            f"Shape: {ps_image.shape}\n"
            f"Voxel size: {voxel_size_formatted}\n"
            f"Type: {ps_image.semantic_type.value}\n"
            f"Layout: {ps_image.image_layout.value}"
        )
        self.widget_info.value = str_info

    def _on_set_voxel_size_layer_done(self):
        logger.debug("_on_set_voxel_size_layer_done called!")

    def _on_info_layer_changed(self, layer):
        if layer is None:
            return

        if not (isinstance(layer, Labels) or isinstance(layer, Image)):
            logger.debug(f"Can't show info for {layer}")
            return

        logger.debug(f"_on_info_layer_changed called for layer {layer}!")

        self.widget_details_layer_select.layer.value = layer
        ps_image = PlantSegImage.from_napari_layer(layer)
        if ps_image.has_valid_voxel_size():
            voxel_size_formatted = "("
            for vs in ps_image.voxel_size:
                voxel_size_formatted += f"{vs:.2f}, "

            voxel_size_formatted = (
                voxel_size_formatted[:-2] + f") {ps_image.voxel_size.unit}"
            )
        else:
            voxel_size_formatted = "None"

        str_info = (
            f"Shape: {ps_image.shape}\n"
            f"Voxel size: {voxel_size_formatted}\n"
            f"Type: {ps_image.semantic_type.value}\n"
            f"Layout: {ps_image.image_layout.value}"
        )
        self.widget_info.value = str_info

    @magic_factory(
        call_button=False,
        title={
            "label": "Details:",
            "widget_type": "EmptyWidget",
        },
        layer={
            "label": "Layer",
            "tooltip": "Select layer to show its details, and change its voxel size.",
        },
    )
    def factory_details_layer_select(
        self,
        title: str = "",
        layer: Image | None = None,
    ):
        pass


class Docs_Container:
    def __init__(self):
        logger.debug("Docs init")
        self.logo_path = (
            Path(__file__).resolve().parent.parent.parent
            / "resources"
            / "logo_white.png"
        )
        assert self.logo_path.exists(), "Logo not found!"
        self.docs_url = "https://kreshuklab.github.io/plant-seg/"

    def get_doc_container(self) -> Container:
        logger.debug("get_doc_container called!")
        """Creates a container with a documentation button and a logo."""

        button = PushButton(text="Open Documentation")
        button.changed.connect(self.open_docs)
        container = Container(
            widgets=[button],
            label=f'<img src="{self.logo_path}">',
            layout="horizontal",
            labels=False,
        )
        container[0].show()
        return Container(widgets=[container], labels=True, layout="horizontal")

    def open_docs(self, button):
        logger.debug("open_docs called!")
        """Open the documentation URL in the default web browser when the button is clicked."""
        webbrowser.open(self.docs_url, new=0, autoraise=True)
        logger.info(f"Docs webpage opened: {self.docs_url}")
        return button
