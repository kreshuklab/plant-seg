import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path

from magicgui import magicgui
from magicgui.widgets import Label
from napari.layers import Image, Labels, Layer
from napari.types import LayerDataTuple

from plantseg.core.image import ImageLayout, ImageType, PlantSegImage, SemanticType
from plantseg.io import H5_EXTENSIONS, ZARR_EXTENSIONS
from plantseg.io.h5 import list_h5_keys
from plantseg.io.zarr import list_zarr_keys
from plantseg.tasks.io_tasks import export_image_task, import_image_task
from plantseg.tasks.workflow_handler import workflow_handler
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.utils import _return_value_if_widget, schedule_task

current_dataset_keys: list[str] | None = None


def get_current_dataset_keys(
    widget,  # Required by magicgui. pylint: disable=unused-argument
) -> list[str] | list[None]:
    if current_dataset_keys is None:
        return [None]
    return current_dataset_keys


class PathMode(Enum):
    FILE = "tiff, h5, etc.."
    DIR = "zarr"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


########################################################################################################################
#                                                                                                                      #
# Open File Widget                                                                                                     #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button="Open File",
    path_mode={
        "label": "File type",
        "choices": PathMode.to_choices(),
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
    },
    path={
        "label": "File path\n(tiff, h5, zarr, png, jpg)",
        "mode": "r",
        "tooltip": "Select a file to be imported, the file can be a tiff, h5, png, jpg.",
    },
    new_layer_name={
        "label": "Layer name",
        "tooltip": "Define the name of the output layer, default is either image or label.",
    },
    layer_type={
        "label": "Layer type",
        "tooltip": "Select if the image is a normal image or a segmentation",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": ImageType.to_choices(),
    },
    dataset_key={
        "label": "Key (h5/zarr only)",
        "widget_type": "ComboBox",
        "choices": get_current_dataset_keys,
        "tooltip": "Key to be loaded from h5",
    },
    stack_layout={
        "label": "Stack layout",
        "choices": ImageLayout.to_choices(),
        "tooltip": "Stack layout",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
    },
)
def widget_open_file(
    path_mode: str = PathMode.FILE.value,
    path: Path = Path.home(),
    layer_type: str = ImageType.IMAGE.value,
    new_layer_name: str = "",
    dataset_key: str | None = None,
    stack_layout: str = ImageLayout.ZYX.value,
) -> Future[LayerDataTuple]:
    """Open a file and return a napari layer."""

    if layer_type == ImageType.IMAGE.value:
        semantic_type = SemanticType.RAW
    elif layer_type == ImageType.LABEL.value:
        semantic_type = SemanticType.SEGMENTATION

    return schedule_task(
        import_image_task,
        task_kwargs={
            "input_path": path,
            "key": dataset_key,
            "image_name": new_layer_name,
            "semantic_type": semantic_type,
            "stack_layout": stack_layout,
        },
        widgets_to_update=[],
    )


widget_open_file.dataset_key.hide()


def generate_layer_name(path: Path, dataset_key: str) -> str:
    dataset_key = dataset_key.replace("/", "_")
    return path.stem + dataset_key


def look_up_dataset_keys(path: Path):
    path = _return_value_if_widget(path)
    ext = path.suffix

    if ext in H5_EXTENSIONS:
        widget_open_file.dataset_key.show()
        dataset_keys = list_h5_keys(path)

    elif ext in ZARR_EXTENSIONS:
        widget_open_file.dataset_key.show()
        dataset_keys = list_zarr_keys(path)

    else:
        widget_open_file.new_layer_name.value = generate_layer_name(path, "")
        widget_open_file.dataset_key.hide()
        return

    global current_dataset_keys
    current_dataset_keys = dataset_keys.copy()  # Update the global variable
    widget_open_file.dataset_key.choices = dataset_keys
    if dataset_keys == [None]:
        widget_open_file.dataset_key.hide()
    if widget_open_file.dataset_key.value not in dataset_keys:
        widget_open_file.dataset_key.value = dataset_keys[0]
        widget_open_file.new_layer_name.value = generate_layer_name(path, widget_open_file.dataset_key.value)


@widget_open_file.path_mode.changed.connect
def _on_path_mode_changed(path_mode: str):
    path_mode = _return_value_if_widget(path_mode)
    if path_mode == PathMode.FILE.value:  # file
        widget_open_file.path.mode = "r"
        widget_open_file.path.label = "File path\n(.tiff, .h5, .png, .jpg)"
    elif path_mode == PathMode.DIR.value:  # directory case
        widget_open_file.path.mode = "d"
        widget_open_file.path.label = "Zarr path\n(.zarr)"


@widget_open_file.path.changed.connect
def _on_path_changed(path: Path):
    look_up_dataset_keys(path)


@widget_open_file.dataset_key.changed.connect
def _on_dataset_key_changed(dataset_key: str):
    dataset_key = _return_value_if_widget(dataset_key)
    if dataset_key:
        widget_open_file.new_layer_name.value = generate_layer_name(widget_open_file.path.value, dataset_key)


@widget_open_file.called.connect
def _on_done(*args):  # Required by magicgui. pylint: disable=unused-argument
    look_up_dataset_keys(widget_open_file.path.value)


########################################################################################################################
#                                                                                                                      #
# Export Stack Widget                                                                                                  #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button="Export Layers",
    images={
        "label": "Layers to export",
        "layout": "vertical",
        "tooltip": "Select all layer to be exported, and (optional) set a custom file name suffix that will be "
        "appended at end of the layer name.",
    },
    data_type={
        "label": "Data Type",
        "choices": ["float32", "uint8", "uint16"],
        "tooltip": "Export datatype (uint16 for segmentation) and all others for images.",
    },
    export_format={
        "label": "Export format",
        "choices": ["tiff", "h5", "zarr"],
        "tooltip": "Export format, if tiff is selected, each layer will be exported as a separate file. "
        "If h5 is selected, all layers will be exported in a single file.",
    },
    directory={
        "label": "Directory to export files",
        "mode": "d",
        "tooltip": "Select the directory where the files will be exported",
    },
    workflow_name={
        "label": "Workflow name",
        "tooltip": "Name of the workflow object.",
    },
)
def widget_export_stacks(
    images: list[tuple[Layer, str]],
    directory: Path = Path.home(),
    export_format: str = "tiff",
    rescale_to_original_resolution: bool = True,
    data_type: str = "float32",
    workflow_name: str = "workflow",
) -> None:
    timer = time.time()
    log("export_image_task started", thread="Export stacks", level="info")

    for i, (image, image_custom_name) in enumerate(images):
        # parse and check input to the function

        image_custom_name = f"exported_image_{i}" if image_custom_name == "" else image_custom_name
        if not isinstance(image, (Image, Labels)):
            raise ValueError("Only Image and Labels layers are supported for PlantSeg export.")
        ps_image = PlantSegImage.from_napari_layer(image)

        export_image_task(
            image=ps_image,
            output_directory=directory,
            output_file_name=image_custom_name,
            custom_key_suffix=image.name,
            scale_to_origin=rescale_to_original_resolution,
            file_format=export_format,
            dtype=data_type,
        )

    # Save the workflow as a yaml file
    workflow_path = Path(directory) / f"{workflow_name}.yaml"
    workflow_handler.save_to_yaml(path=workflow_path)

    log(f"export_image_task completed in {time.time() - timer:.2f}s", thread="Export stacks", level="info")


export_details = [
    widget_export_stacks.directory,
    widget_export_stacks.export_format,
    widget_export_stacks.rescale_to_original_resolution,
    widget_export_stacks.data_type,
    widget_export_stacks.workflow_name,
]
for widget in export_details:
    widget.hide()


@widget_export_stacks.images.changed.connect
def _on_images_changed(images_list: list[tuple[Layer, str]]):
    images_list = _return_value_if_widget(images_list)
    if len(images_list) > 0:
        for widget in export_details:
            widget.show()
    else:
        for widget in export_details:
            widget.hide()


########################################################################################################################
#                                                                                                                      #
# Show Image Infos Widget                                                                                              #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button=None,
    auto_call=True,
    layer={
        "label": "Select layer",
        "tooltip": "Select the image or label to show the information.",
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_show_info(layer: Layer, update_other_widgets: bool = False) -> None:
    """Show the information of the selected layer."""


widget_infos = Label(
    value="Select layer to show information here...", label="Infos", tooltip="Information about the selected layer."
)


@widget_show_info.layer.changed.connect
def _on_layer_changed(layer):
    if not (isinstance(layer, Labels) or isinstance(layer, Image)):
        raise ValueError("Info can only be shown for Image or Labels layers.")

    ps_image = PlantSegImage.from_napari_layer(layer)
    if ps_image.has_valid_voxel_size():
        voxel_size_formatted = "("
        for vs in ps_image.voxel_size:
            voxel_size_formatted += f"{vs:.2f}, "

        voxel_size_formatted = voxel_size_formatted[:-2] + f") {ps_image.voxel_size.unit}"
    else:
        voxel_size_formatted = "None"

    str_info = (
        f"Shape: {ps_image.shape}, "
        f"Voxel size: {voxel_size_formatted}, "
        f"Type: {ps_image.semantic_type.value}, "
        f"Layout: {ps_image.image_layout.value}"
    )
    widget_infos.value = str_info
