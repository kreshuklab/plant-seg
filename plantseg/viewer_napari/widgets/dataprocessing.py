from concurrent.futures import Future
from enum import Enum

from magicgui import magicgui
from napari.layers import Image, Labels, Layer
from napari.types import LayerDataTuple

from plantseg.core.image import PlantSegImage
from plantseg.core.voxelsize import VoxelSize
from plantseg.core.zoo import model_zoo
from plantseg.tasks.dataprocessing_tasks import (
    gaussian_smoothing_task,
    image_rescale_to_shape_task,
    image_rescale_to_voxel_size_task,
    remove_false_positives_by_foreground_probability_task,
    set_voxel_size_task,
)
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.utils import schedule_task

########################################################################################################################
#                                                                                                                      #
# Gaussian Smoothing Widget                                                                                            #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button="Run Gaussian Smoothing",
    image={
        "label": "Image",
        "tooltip": "Image layer to apply the smoothing.",
    },
    sigma={
        "label": "Sigma",
        "widget_type": "FloatSlider",
        "tooltip": "Define the size of the gaussian smoothing kernel. "
        "The larger the more blurred will be the output image.",
        "max": 10.0,
        "min": 0.1,
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_gaussian_smoothing(
    image: Image, sigma: float = 1.0, update_other_widgets: bool = True
) -> Future[LayerDataTuple]:
    """Apply Gaussian smoothing to an image layer."""

    ps_image = PlantSegImage.from_napari_layer(image)

    widgets_to_update = []  # TODO

    return schedule_task(
        gaussian_smoothing_task,
        task_kwargs={
            "image": ps_image,
            "sigma": sigma,
        },
        widgets_to_update=widgets_to_update if update_other_widgets else [],
    )


########################################################################################################################
#                                                                                                                      #
# Rescaling Widget                                                                                                     #
#                                                                                                                      #
########################################################################################################################


class RescaleType(Enum):
    NEAREST = (0, "Nearest")
    LINEAR = (1, "Linear")
    BILINEAR = (2, "Bilinear")

    def __init__(self, int_val, str_val):
        self.int_val = int_val
        self.str_val = str_val

    @classmethod
    def to_choices(cls):
        return [(mode.str_val, mode.int_val) for mode in cls]


class RescaleModes(Enum):
    FROM_FACTOR = "From factor"
    TO_LAYER_VOXEL_SIZE = "To layer voxel size"
    TO_LAYER_SHAPE = "To layer shape"
    TO_MODEL_VOXEL_SIZE = "To model voxel size"
    TO_VOXEL_SIZE = "To voxel size"
    TO_SHAPE = "To shape"
    SET_VOXEL_SIZE = "Set voxel size"

    @classmethod
    def to_choices(cls):
        return [(mode.value, mode) for mode in RescaleModes]


@magicgui(
    call_button="Run Rescaling",
    image={
        "label": "Image or Label",
        "tooltip": "Layer to apply the rescaling.",
    },
    mode={
        "label": "Rescale mode",
        "choices": RescaleModes.to_choices(),
    },
    rescaling_factor={
        "label": "Rescaling factor",
        "tooltip": "Define the scaling factor to use for resizing the input image.",
        "options": {"step": 0.0001},
    },
    out_voxel_size={
        "label": "Out voxel size",
        "tooltip": "Define the output voxel size. Units are same as imported, "
        '(if units are missing default is "um").',
        "options": {"step": 0.0001},
    },
    reference_layer={
        "label": "Reference layer",
        "tooltip": "Rescale to same voxel size as selected layer.",
    },
    reference_model={
        "label": "Reference model",
        "tooltip": "Rescale to same voxel size as selected model.",
        "choices": model_zoo.list_models(),
    },
    reference_shape={
        "label": "Out shape",
        "tooltip": "Rescale to a manually selected shape.",
    },
    order={
        "label": "Interpolation order",
        "widget_type": "ComboBox",
        "choices": RescaleType.to_choices(),
        "tooltip": "0 for nearest neighbours (default for labels), 1 for linear, 2 for bilinear.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_rescaling(
    image: Layer,
    mode: RescaleModes = RescaleModes.FROM_FACTOR,
    rescaling_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    out_voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_layer: Layer | None = None,
    reference_model: str = model_zoo.list_models()[0],
    reference_shape: tuple[int, int, int] = (1, 1, 1),
    order: int = 0,
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    """Rescale an image or label layer."""

    if isinstance(image, Image) or isinstance(image, Labels):
        ps_image = PlantSegImage.from_napari_layer(image)
    else:
        raise ValueError("Image must be an Image or Label layer.")

    # Cover set voxel size case
    if not ps_image.has_valid_original_voxel_size():
        if mode not in [RescaleModes.SET_VOXEL_SIZE, RescaleModes.TO_LAYER_SHAPE, RescaleModes.TO_SHAPE]:
            raise ValueError("Original voxel size is missing, please set the voxel size manually.")

    # TODO add list of widgets to update
    widgets_to_update = []

    if mode == RescaleModes.SET_VOXEL_SIZE:
        # Run set voxel size task
        return schedule_task(
            set_voxel_size_task,
            task_kwargs={
                "image": ps_image,
                "voxel_size": out_voxel_size,
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )

    if mode in [RescaleModes.TO_LAYER_SHAPE, RescaleModes.TO_SHAPE]:
        if mode == RescaleModes.TO_LAYER_SHAPE:
            output_shape = reference_layer.data.shape

        if mode == RescaleModes.TO_SHAPE:
            output_shape = reference_shape

        return schedule_task(
            image_rescale_to_shape_task,
            task_kwargs={
                "image": ps_image,
                "new_shape": output_shape,
                "order": order,
            },
            widgets_to_update=widgets_to_update,
        )

    # Cover rescale that requires a valid voxel size
    current_voxel_size = ps_image.voxel_size
    if mode == RescaleModes.FROM_FACTOR:
        out_voxel_size = current_voxel_size.voxelsize_from_factor(rescaling_factor)

    if mode == RescaleModes.TO_VOXEL_SIZE:
        out_voxel_size = VoxelSize(voxels_size=out_voxel_size, unit=current_voxel_size.unit)

    if mode == RescaleModes.TO_LAYER_VOXEL_SIZE:
        if not (isinstance(reference_layer, Image) or isinstance(reference_layer, Labels)):
            raise ValueError("Reference layer must be an Image or Label layer.")
        reference_ps_image = PlantSegImage.from_napari_layer(reference_layer)
        out_voxel_size = reference_ps_image.voxel_size

    if mode == RescaleModes.TO_MODEL_VOXEL_SIZE:
        model_voxel_size = model_zoo.get_model_resolution(reference_model)
        if model_voxel_size is None:
            raise ValueError(f"Model {reference_model} does not have a resolution defined.")
        out_voxel_size = VoxelSize(voxels_size=model_voxel_size, unit=current_voxel_size.unit)

    return schedule_task(
        image_rescale_to_voxel_size_task,
        task_kwargs={
            "image": ps_image,
            "new_voxel_size": out_voxel_size,
            "order": order,
        },
        widgets_to_update=widgets_to_update,
    )


list_widget_rescaling_all = [
    widget_rescaling.out_voxel_size,
    widget_rescaling.reference_layer,
    widget_rescaling.reference_model,
    widget_rescaling.rescaling_factor,
    widget_rescaling.reference_shape,
]

for widget in list_widget_rescaling_all:
    widget.hide()
widget_rescaling.rescaling_factor.show()
widget_rescaling.reference_shape[0].max = 20000
widget_rescaling.reference_shape[1].max = 20000
widget_rescaling.reference_shape[2].max = 20000


@widget_rescaling.mode.changed.connect
def _rescale_update_visibility(mode: RescaleModes):
    for widget in list_widget_rescaling_all:
        widget.hide()

    match mode:
        case RescaleModes.FROM_FACTOR:
            widget_rescaling.rescaling_factor.show()

        case RescaleModes.TO_LAYER_VOXEL_SIZE:
            widget_rescaling.reference_layer.show()

        case RescaleModes.TO_MODEL_VOXEL_SIZE:
            widget_rescaling.reference_model.show()

        case RescaleModes.TO_VOXEL_SIZE:
            widget_rescaling.out_voxel_size.show()

        case RescaleModes.TO_LAYER_SHAPE:
            widget_rescaling.reference_layer.show()

        case RescaleModes.TO_SHAPE:
            widget_rescaling.reference_shape.show()

        case RescaleModes.SET_VOXEL_SIZE:
            widget_rescaling.out_voxel_size.show()

        case _:
            raise ValueError(f"{mode} is not implemented yet.")


list_widget_rescaling_3d = [
    widget_rescaling.rescaling_factor[0],
    widget_rescaling.reference_shape[0],
    widget_rescaling.out_voxel_size[0],
]


@widget_rescaling.image.changed.connect
def _on_rescaling_image_changed(image: Layer):
    if not (isinstance(image, Image) or isinstance(image, Labels)):
        raise ValueError("Image must be an Image or Label layer.")

    if image.data.ndim == 2 or (image.data.ndim == 3 and image.data.shape[0] == 1):
        for widget in list_widget_rescaling_3d:
            widget.hide()
    else:
        for widget in list_widget_rescaling_3d:
            widget.show()

    offset = 1 if image.data.ndim == 2 else 0
    for i, (shape, scale) in enumerate(zip(image.data.shape, image.scale)):
        widget_rescaling.out_voxel_size[i + offset].value = scale
        widget_rescaling.reference_shape[i + offset].value = shape

    if isinstance(image, Labels):
        widget_rescaling.order.value = RescaleType.NEAREST.int_val


@widget_rescaling.order.changed.connect
def _on_rescale_order_changed(order):
    current_image = widget_rescaling.image.value

    if current_image is None:
        return None

    if isinstance(current_image, Labels) and order != RescaleType.NEAREST.int_val:
        log(
            "Labels can only be rescaled with nearest interpolation",
            thread="Rescaling",
            level="warning",
        )
        widget_rescaling.order.value = RescaleType.NEAREST.int_val


@magicgui(
    call_button="Remove False Positives",
    segmentation={
        "label": "Segmentation",
        "tooltip": "Segmentation layer to remove false positives.",
    },
    foreground={
        "label": "Foreground",
        "tooltip": "Foreground probability layer.",
    },
    threshold={
        "label": "Threshold",
        "tooltip": "Threshold value to remove false positives.",
        'widget_type': 'FloatSlider',
        "max": 1.0,
        "min": 0.0,
        "step": 0.01,
    },
)
def widget_remove_false_positives_by_foreground(
    segmentation: Labels, foreground: Image, threshold: float = 0.5
) -> Future[LayerDataTuple]:
    """Remove false positives from a segmentation layer using a foreground probability layer."""

    ps_segmentation = PlantSegImage.from_napari_layer(segmentation)
    ps_foreground = PlantSegImage.from_napari_layer(foreground)

    return schedule_task(
        remove_false_positives_by_foreground_probability_task,
        task_kwargs={
            "segmentation": ps_segmentation,
            "foreground": ps_foreground,
            "threshold": threshold,
        },
        widgets_to_update=[],
    )
