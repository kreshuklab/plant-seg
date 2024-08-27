import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import List, Tuple

from magicgui import magicgui
from magicgui.widgets import Label
from napari.layers import Image, Labels, Layer
from napari.types import LayerDataTuple

from plantseg.io import H5_EXTENSIONS, ZARR_EXTENSIONS
from plantseg.io.h5 import list_h5_keys
from plantseg.io.zarr import list_zarr_keys
from plantseg.plantseg_image import ImageLayout, ImageType, PlantSegImage, SemanticType
from plantseg.tasks.io_tasks import export_image_task, import_image_task
from plantseg.tasks.workflow_handler import workflow_handler
from plantseg.viewer_napari import napari_formatted_logging
from plantseg.viewer_napari.widgets.utils import _return_value_if_widget, schedule_task


class PathMode(Enum):
    FILE = "tiff, h5, etc.."
    DIR = "zarr"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


@magicgui(
    call_button="Open file",
    path_mode={
        "label": "File type",
        "choices": PathMode.to_choices(),
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
    },
    path={
        "label": "Pick a file (tiff, h5, zarr, png, jpg)",
        "mode": "r",
        "tooltip": "Select a file to be imported, the file can be a tiff, h5, png, jpg.",
    },
    new_layer_name={
        "label": "Layer Name",
        "tooltip": "Define the name of the output layer, default is either image or label.",
    },
    layer_type={
        "label": "Layer Type",
        "tooltip": "Select if the image is a normal image or a segmentation",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": ImageType.to_choices(),
    },
    key={
        "label": "Key (h5/zarr only)",
        "choices": [""],
        "tooltip": "Key to be loaded from h5",
    },
    stack_layout={
        "label": "Stack Layout",
        "choices": ImageLayout.to_choices(),
        "tooltip": "Stack layout",
    },
)
def widget_open_file(
    path_mode: str = PathMode.FILE.value,
    path: Path = Path.home(),
    layer_type: str = ImageType.IMAGE.value,
    new_layer_name: str = "",
    key: str = "",
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
            "key": key,
            "image_name": new_layer_name,
            "semantic_type": semantic_type,
            "stack_layout": stack_layout,
        },
        widget_to_update=[],
    )


widget_open_file.key.hide()


@widget_open_file.path_mode.changed.connect
def _on_path_mode_changed(path_mode: str):
    path_mode = _return_value_if_widget(path_mode)
    if path_mode == PathMode.FILE.value:  # file
        widget_open_file.path.mode = "r"
        widget_open_file.path.label = "Pick a file (.tiff, .h5, .png, .jpg)"
    elif path_mode == PathMode.DIR.value:  # directory case
        widget_open_file.path.mode = "d"
        widget_open_file.path.label = "Pick a folder (.zarr)"


@widget_open_file.path.changed.connect
def _on_path_changed(path: Path):
    path = _return_value_if_widget(path)
    widget_open_file.new_layer_name.value = path.stem
    ext = path.suffix

    if ext in H5_EXTENSIONS:
        widget_open_file.key.show()
        keys = list_h5_keys(path)
        widget_open_file.key.choices = keys
        widget_open_file.key.value = keys[0]

    elif ext in ZARR_EXTENSIONS:
        widget_open_file.key.show()
        keys = list_zarr_keys(path)
        widget_open_file.key.choices = keys
        widget_open_file.key.value = keys[0]


# For some reason after the widget is called the keys are deleted, so we need to reassign them after the widget is called
@widget_open_file.called.connect
def _on_done(*args):
    _on_path_changed(widget_open_file.path.value)


@magicgui(
    call_button="Export stack",
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
    images: List[Tuple[Layer, str]],
    directory: Path = Path.home(),
    export_format: str = "tiff",
    rescale_to_original_resolution: bool = True,
    data_type: str = "float32",
    workflow_name: str = "workflow",
) -> None:
    timer = time.time()
    napari_formatted_logging("export_image_task started", thread="Task", level="info")

    for i, (image, image_custom_name) in enumerate(images):
        # parse and check input to the function

        image_custom_name = f"exported_image_{i}" if image_custom_name == "" else image_custom_name
        ps_image = PlantSegImage.from_napari_layer(image)

        export_image_task(
            image=ps_image,
            output_directory=directory,
            output_file_name=image_custom_name,
            custom_key=image.name,
            scale_to_origin=rescale_to_original_resolution,
            file_format=export_format,
            dtype=data_type,
        )

    # Save the workflow as a yaml file
    workflow_path = Path(directory) / f"{workflow_name}.yaml"
    workflow_handler.save_to_yaml(path=workflow_path)

    napari_formatted_logging(f"export_image_task completed in {time.time() - timer:.2f}s", thread="Task", level="info")


widget_export_stacks.directory.hide()
widget_export_stacks.export_format.hide()
widget_export_stacks.rescale_to_original_resolution.hide()
widget_export_stacks.data_type.hide()
widget_export_stacks.workflow_name.hide()


@widget_export_stacks.images.changed.connect
def _on_images_changed(images_list: List[Tuple[Layer, str]]):
    images_list = _return_value_if_widget(images_list)
    if len(images_list) > 0:
        widget_export_stacks.directory.show()
        widget_export_stacks.export_format.show()
        widget_export_stacks.rescale_to_original_resolution.show()
        widget_export_stacks.data_type.show()
        widget_export_stacks.workflow_name.show()
    else:
        widget_export_stacks.directory.hide()
        widget_export_stacks.export_format.hide()
        widget_export_stacks.rescale_to_original_resolution.hide()
        widget_export_stacks.data_type.hide()
        widget_export_stacks.workflow_name.hide()


########################################################################################################################
#                                                                                                                      #
# Show Image Infos                                                                                                     #
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
        for vs in ps_image.voxel_size.voxels_size:
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
