from concurrent.futures import Future
from enum import Enum

from magicgui import magicgui
from napari.layers import Image, Labels, Layer, Shapes
from napari.types import LayerDataTuple

from plantseg.core.image import PlantSegImage
from plantseg.core.voxelsize import VoxelSize
from plantseg.core.zoo import model_zoo
from plantseg.tasks.dataprocessing_tasks import (
    fix_over_under_segmentation_from_nuclei_task,
    gaussian_smoothing_task,
    image_cropping_task,
    image_rescale_to_shape_task,
    image_rescale_to_voxel_size_task,
    remove_false_positives_by_foreground_probability_task,
    set_voxel_size_task,
    relabel_segmentation_task,
    set_biggest_instance_to_zero_task,
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
# Cropping Widget                                                                                                      #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button=f"Run Cropping",
    image={
        "label": "Image or Label",
        "tooltip": "Layer to apply the rescaling.",
    },
    crop_roi={
        "label": "Crop ROI",
        "tooltip": "This must be a shape layer with a rectangle XY overlaying the area to crop.",
    },
    crop_z={
        "label": "Z slices [Start, End)",
        "tooltip": "Number of z slices to take next to the current selection.\nSTART is included, END is not.",
        "widget_type": "RangeSlider",
        "max": 100,
        "min": 0,
        "step": 1,
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_cropping(
    image: Layer,
    crop_roi: Shapes | None = None,
    crop_z: tuple[int, int] = (0, 100),
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    if crop_roi is not None:
        assert len(crop_roi.shape_type) == 1, "Only one rectangle should be used for cropping"
        assert crop_roi.shape_type[0] == "rectangle", "Only a rectangle shape should be used for cropping"

    if not isinstance(image, (Image, Labels)):
        raise ValueError(f"{type(image)} cannot be cropped, please use Image layers or Labels layers")

    if crop_roi is not None:
        rectangle = crop_roi.data[0].astype("int64")
    else:
        rectangle = None

    ps_image = PlantSegImage.from_napari_layer(image)

    widgets_to_update = []

    return schedule_task(
        image_cropping_task,
        task_kwargs={
            "image": ps_image,
            "rectangle": rectangle,
            "crop_z": crop_z,
        },
        widgets_to_update=widgets_to_update if update_other_widgets else [],
    )


@widget_cropping.image.changed.connect
def _on_cropping_image_changed(image: Layer):
    if image is None:
        widget_cropping.crop_z.hide()
        return None

    image_shape_z = int(image.data.shape[0])

    if image_shape_z == 1:
        widget_cropping.crop_z.hide()
        return None

    widget_cropping.crop_z.show()
    widget_cropping.crop_z.step = 1

    if widget_cropping.crop_z.value[1] > image_shape_z:
        widget_cropping.crop_z.value = (0, image_shape_z)
    widget_cropping.crop_z.max = image_shape_z


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
        "label": "Select layer",
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

    # TODO: Write tests for 2D images and labels, then change `.enabled` to `.hide()`
    if image.data.ndim == 2 or (image.data.ndim == 3 and image.data.shape[0] == 1):
        for widget in list_widget_rescaling_3d:
            widget.enabled = False
    else:
        for widget in list_widget_rescaling_3d:
            widget.enabled = True

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


########################################################################################################################
#                                                                                                                      #
# Remove False Positives Widget                                                                                        #
#                                                                                                                      #
########################################################################################################################


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


########################################################################################################################
#                                                                                                                      #
# Fix Over-/Under-Segmentation Widget                                                                                  #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button='Fix Segmentation by Nuclei',
    segmentation_cells={'label': 'Cell instances'},
    segmentation_nuclei={'label': 'Nuclear instances'},
    boundary_pmaps={'label': 'Boundary Pmap/Image'},
    threshold={
        'label': 'Threshold',
        'widget_type': 'FloatRangeSlider',
        'max': 100,
        'min': 0,
        'step': 0.1,
    },
    quantile={
        'label': 'Nuclei Quantile',
        'widget_type': 'FloatRangeSlider',
        'max': 100,
        'min': 0,
        'step': 0.1,
    },
)
def widget_fix_over_under_segmentation_from_nuclei(
    segmentation_cells: Labels,
    segmentation_nuclei: Labels,
    boundary_pmaps: Image | None = None,
    threshold=(33, 66),
    quantile=(0.3, 99.9),
) -> Future[LayerDataTuple]:
    """
    Widget interface for correcting over- and under-segmentation of cells based on nuclei segmentation.

    This GUI interface allows the user to specify the input cell and nuclear segmentations, along with optional boundary
    probability maps. The user can control the merging and splitting thresholds, and define quantiles to filter out
    irregular nuclei. The widget schedules the correction task in the background and updates the displayed results accordingly.

    Args:
        cell_segmentation (Labels): Input label layer for cell segmentation.
        nuclei_segmentation (Labels): Input label layer for nuclei segmentation.
        boundary_pmaps (Image | None, optional): Optional boundary probability map or image to assist in segmentation refinement.
        threshold (tuple[float, float], optional): Threshold range for merging (first value) and splitting (second value) cells.
            The values should be between 0 and 100, corresponding to 0%-100% overlap. Default is (33, 66).
        quantile (tuple[float, float], optional): Quantile range to filter nuclei size, ignoring outliers.
            Values should be between 0 and 100. Default is (0.3, 99.9).

    Returns:
        Future[LayerDataTuple]: A future object that contains the corrected segmentation layer once the task completes.
    """
    ps_seg_cel = PlantSegImage.from_napari_layer(segmentation_cells)
    ps_seg_nuc = PlantSegImage.from_napari_layer(segmentation_nuclei)
    if boundary_pmaps:
        ps_pmap_cell_boundary = PlantSegImage.from_napari_layer(boundary_pmaps)
    else:
        ps_pmap_cell_boundary = None
    threshold_merge, threshold_split = threshold[0] / 100, threshold[1] / 100

    return schedule_task(
        fix_over_under_segmentation_from_nuclei_task,
        task_kwargs={
            'cell_seg': ps_seg_cel,
            'nuclei_seg': ps_seg_nuc,
            'threshold_merge': threshold_merge,
            'threshold_split': threshold_split,
            'quantiles_nuclei': quantile,
            'boundary': ps_pmap_cell_boundary,
        },
        widgets_to_update=[],
    )


########################################################################################################################
#                                                                                                                      #
# Relabel Widget                                                                                                       #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button=f"Relabel Instances",
    segmentation={
        "label": "Segmentation",
        "tooltip": "Segmentation can be any label layer.",
    },
)
def widget_label_processing(
    segmentation: Labels,
) -> Future[LayerDataTuple]:
    """Relabel an image layer."""

    ps_image = PlantSegImage.from_napari_layer(segmentation)

    return schedule_task(
        relabel_segmentation_task,
        task_kwargs={
            "image": ps_image,
        },
        widgets_to_update=[],
    )


########################################################################################################################
#                                                                                                                      #
# Set Biggest Instance to Zero Widget                                                                                  #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button=f"Set Biggest Instance to Zero",
    segmentation={
        "label": "Segmentation",
        "tooltip": "Segmentation can be any label layer.",
    },
)
def widget_set_biggest_instance_to_zero(
    segmentation: Labels,
) -> Future[LayerDataTuple]:
    """Set the biggest instance to zero in a label layer."""

    ps_image = PlantSegImage.from_napari_layer(segmentation)

    return schedule_task(
        set_biggest_instance_to_zero_task,
        task_kwargs={
            "image": ps_image,
        },
        widgets_to_update=[],
    )
