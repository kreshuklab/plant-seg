import time
from pathlib import Path

from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image, Labels, Layer
from psygnal import Signal

from plantseg.core.image import PlantSegImage
from plantseg.tasks.io_tasks import export_image_task
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.utils import div


class Output_Tab:
    successful_export = Signal()

    def __init__(self):
        self.widget_export_image = self.factory_export_image()
        self.widget_export_image.self.bind(self)

        self.export_details = [
            self.widget_export_image.directory,
            self.widget_export_image.name_pattern,
            self.widget_export_image.export_format,
            self.widget_export_image.data_type,
            self.widget_export_image.scale_to_origin,
            self.widget_export_image.key,
        ]
        self._toggle_export_details_widgets(False)
        self._toggle_key(False)
        self.widget_export_image.image.changed.connect(self._on_images_changed)
        self.widget_export_image.export_format.changed.connect(
            self._on_export_format_changed
        )

    def get_container(self):
        return Container(
            widgets=[
                div("Output"),
                self.widget_export_image,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Export Layer",
        image={
            "label": "Layer to export",
            "tooltip": "Select all layer to be exported, and (optional) set a custom file name suffix that will be "
            "appended at end of the layer name.",
        },
        directory={
            "label": "Export directory",
            "mode": "d",
            "tooltip": "Select the directory where the files will be exported",
        },
        name_pattern={
            "label": "Export name pattern",
            "tooltip": "Pattern for the exported file name. Use {image_name} to include the layer name, "
            "or {file_name} to include the original file name.",
        },
        export_format={
            "label": "Export format",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
            "choices": ["tiff", "h5", "zarr"],
            "tooltip": "Export format, if tiff is selected, each layer will be exported as a separate file. "
            "If h5 is selected, all layers will be exported in a single file.",
        },
        key={
            "label": "Key",
            "tooltip": "Key to be used in the h5 or zarr file.",
        },
        scale_to_origin={
            "label": "Rescale to original resolution",
            "tooltip": "Rescale the image to the original voxel size.",
        },
        data_type={
            "label": "Data Type",
            "choices": ["float32", "uint8", "uint16"],
            "tooltip": "Export datatype (uint16 for segmentation) and all others for images.",
        },
    )
    def factory_export_image(
        self,
        image: Layer | None = None,
        directory: Path = Path.home(),
        name_pattern: str = "{file_name}_export",
        export_format: str = "tiff",
        key: str = "raw",
        scale_to_origin: bool = True,
        data_type: str = "uint16",
    ) -> None | bool:
        """Export layers in various formats."""

        if not isinstance(image, (Image, Labels)):
            log(
                "Please select an Image or Labels layers to export!",
                thread="Output",
                level="WARNING",
            )
            return

        timer = time.time()
        log("export_image_task started", thread="Output", level="info")
        ps_image = PlantSegImage.from_napari_layer(image)

        export_image_task(
            image=ps_image,
            export_directory=directory,
            name_pattern=name_pattern,
            key=key,
            scale_to_origin=scale_to_origin,
            export_format=export_format,
            data_type=data_type,
        )
        timer = time.time() - timer
        log(
            f"export_image_task finished in {timer:.2f} seconds",
            thread="Output",
            level="info",
        )
        self.successful_export.emit(True)

        return

    def _toggle_export_details_widgets(self, show: bool):
        for widget in self.export_details:
            if show:
                widget.show()
            else:
                widget.hide()

    def _toggle_key(self, show: bool):
        if not show or self.widget_export_image.export_format.value == "tiff":
            self.widget_export_image.key.hide()
            return
        self.widget_export_image.key.show()

    def _on_images_changed(self, image: Image | Labels | None):
        if image is None:
            self._toggle_export_details_widgets(False)
            self._toggle_key(False)
            return

        if isinstance(image, Labels) or isinstance(image, Image):
            self._toggle_export_details_widgets(True)
            self._toggle_key(True)
            self.widget_export_image.key.value = (
                "raw" if isinstance(image, Image) else "segmentation"
            )

            if (
                isinstance(image, Labels)
                and self.widget_export_image.data_type.value == "float32"
            ):
                log(
                    "Data type float32 is not supported for Labels layers, changing to uint16",
                    thread="Export stacks",
                    level="warning",
                )
                self.widget_export_image.data_type.value = "uint16"
            return

        raise ValueError(
            "Only Image and Labels layers are supported for PlantSeg export."
        )

    def _on_export_format_changed(self, export_format: str):
        self._toggle_key(export_format != "tiff")
