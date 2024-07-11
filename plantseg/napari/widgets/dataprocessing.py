from concurrent.futures import Future
from enum import Enum

from magicgui import magicgui
from napari.types import LayerDataTuple

from napari.layers import Image
from plantseg.workflows.general_tasks import gaussian_smoothing_task
from plantseg.napari.widgets.utils import schedule_task
from plantseg.image import PlantSegImage


class WidgetName(Enum):
    RESCALING = ("Rescaling", "Rescaled")
    SMOOTHING = ("Gaussian Smoothing", "Smoothed")
    CROPPING = ("Cropping", "Cropped")
    MERGING = ("Layer Merging", None)  # Merged image has special layer names
    CLEANING_LABEL = ("Label Cleaning", "Cleaned")

    def __init__(self, step_name, layer_suffix):
        self.step_name = step_name
        self.layer_suffix = layer_suffix


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
    return schedule_task(
        gaussian_smoothing_task,
        task_kwargs={
            "image": ps_image,
            "sigma": sigma,
        },
        widget_to_update=[],
    )


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
