from concurrent.futures import Future
from enum import Enum
from typing import Union, Optional

import numpy as np
from magicgui import magicgui

import napari  # Napari import is lazy.
from napari.layers import Image, Labels, Shapes, Layer
from napari.types import LayerDataTuple

from plantseg.dataprocessing.functional import image_gaussian_smoothing, image_rescale
from plantseg.dataprocessing.functional.dataprocessing import compute_scaling_factor, compute_scaling_voxelsize
from plantseg.dataprocessing.functional.labelprocessing import relabel_segmentation as _relabel_segmentation
from plantseg.dataprocessing.functional.labelprocessing import set_background_to_value
from plantseg.viewer.widget.predictions import widget_unet_predictions
from plantseg.viewer.widget.segmentation import widget_agglomeration, widget_lifted_multicut, widget_dt_ws
from plantseg.viewer.widget.utils import (
    return_value_if_widget,
    start_threading_process,
    create_layer_name,
    layer_properties,
    napari_formatted_logging,
)
from plantseg.models.zoo import model_zoo


class WidgetName(Enum):
    RESCALING = ("Rescaling", "Rescaled")
    SMOOTHING = ("Gaussian Smoothing", "Smoothed")
    CROPPING = ("Cropping", "Cropped")
    MERGING = ("Layer Merging", None)  # Merged image has special layer names
    CLEANING_LABEL = ("Label Cleaning", "Cleaned")

    def __init__(self, step_name, layer_suffix):
        self.step_name = step_name
        self.layer_suffix = layer_suffix


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
    viewer: napari.Viewer,
    image: Image,
    sigma: float = 1.0,
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    """Apply Gaussian smoothing to an image layer."""
    out_name = create_layer_name(image.name, WidgetName.SMOOTHING.layer_suffix)
    inputs_kwarg = {"image": image.data}
    step_kwargs = {"sigma": sigma}
    inputs_names = (image.name,)
    layer_kwargs = layer_properties(name=out_name, scale=image.scale, metadata=image.metadata)
    layer_type = "image"

    return start_threading_process(
        image_gaussian_smoothing,
        runtime_kwargs=inputs_kwarg,
        statics_kwargs=step_kwargs,
        out_name=out_name,
        input_keys=inputs_names,
        layer_kwarg=layer_kwargs,
        layer_type=layer_type,
        step_name=WidgetName.SMOOTHING.step_name,
        viewer=viewer,
        widgets_to_update=(
            [
                widget_unet_predictions.image,
                widget_agglomeration.image,
                widget_lifted_multicut.image,
                widget_dt_ws.image,
                widget_rescaling.image,
                widget_cropping.image,
            ]
            if update_other_widgets
            else None
        ),
    )


@magicgui(
    call_button=f"Run {WidgetName.RESCALING.step_name}",
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
        "options": {"step": 0.00001},
    },
    out_voxel_size={
        "label": "Out voxel size",
        "tooltip": "Define the output voxel size. Units are same as imported, "
        '(if units are missing default is "um").',
        "options": {"step": 0.00001},
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
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_rescaling(
    viewer: napari.Viewer,
    image: Layer,
    mode: RescaleModes = RescaleModes.FROM_FACTOR,
    rescaling_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    out_voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_layer: Union[Layer, None] = None,
    reference_model: str = model_zoo.list_models()[0],
    reference_shape: tuple[int, int, int] = (1, 1, 1),
    order: int = 0,
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    """Rescale an image or label layer to a new voxel size or shape."""

    if isinstance(image, Image):
        layer_type = "image"

    elif isinstance(image, Labels):
        layer_type = "labels"
        order = 0

    else:
        raise ValueError(f"{type(image)} cannot be rescaled, please use Image layers or Labels layers")

    assert isinstance(image.data, np.ndarray), "Only numpy arrays are supported for rescaling."

    if image.data.ndim == 2:
        rescaling_factor = (1.0,) + rescaling_factor[1:]
    assert (
        len(rescaling_factor) == 3
    ), "Rescaling factor must be a tuple of 3 elements. Please submit an issue on GitHub."
    rescaling_factor = float(rescaling_factor[0]), float(rescaling_factor[1]), float(rescaling_factor[2])

    current_resolution = image.scale
    match mode:
        case RescaleModes.FROM_FACTOR:
            out_voxel_size = compute_scaling_voxelsize(current_resolution, scaling_factor=rescaling_factor)

        case RescaleModes.TO_LAYER_VOXEL_SIZE:
            if reference_layer is None:
                raise ValueError("Please select a reference layer to rescale to.")

            out_voxel_size = reference_layer.scale
            rescaling_factor = compute_scaling_factor(current_resolution, out_voxel_size)

        case RescaleModes.TO_MODEL_VOXEL_SIZE:
            model_voxel_size = model_zoo.get_model_resolution(reference_model)
            if model_voxel_size is None:
                raise ValueError(f"Model {reference_model} does not have a resolution defined.")

            rescaling_factor = compute_scaling_factor(current_resolution, model_voxel_size)
            out_voxel_size = model_voxel_size

        case RescaleModes.TO_VOXEL_SIZE:
            rescaling_factor = compute_scaling_factor(current_resolution, out_voxel_size)

        case RescaleModes.TO_LAYER_SHAPE:
            if reference_layer is None:
                raise ValueError("Please select a reference layer to rescale to.")
            current_shape = image.data.shape
            out_shape = reference_layer.data.shape
            assert len(out_shape) == 3, "Reference layer must be a 3D layer. Please submit an issue on GitHub."
            assert len(current_shape) == 3, "Current layer must be a 3D layer. Please submit an issue on GitHub."
            rescaling_factor = (
                out_shape[0] / current_shape[0],
                out_shape[1] / current_shape[1],
                out_shape[2] / current_shape[2],
            )
            out_voxel_size = (
                current_resolution[0] / rescaling_factor[0],
                current_resolution[1] / rescaling_factor[1],
                current_resolution[2] / rescaling_factor[2],
            )

        case RescaleModes.TO_SHAPE:
            current_shape = image.data.shape
            out_shape = reference_shape
            assert len(out_shape) == 3, "Reference layer must be a 3D layer. Please submit an issue on GitHub."
            assert len(current_shape) == 3, "Current layer must be a 3D layer. Please submit an issue on GitHub."
            rescaling_factor = (
                out_shape[0] / current_shape[0],
                out_shape[1] / current_shape[1],
                out_shape[2] / current_shape[2],
            )
            out_voxel_size = (
                current_resolution[0] / rescaling_factor[0],
                current_resolution[1] / rescaling_factor[1],
                current_resolution[2] / rescaling_factor[2],
            )

        # This is the only case where we don't need to rescale the image data
        # we just need to update the metadata, no need to add this to the DAG.
        # Maybe this will change in the future implementation of the headless mode.
        case RescaleModes.SET_VOXEL_SIZE:
            out_voxel_size = float(out_voxel_size[0]), float(out_voxel_size[1]), float(out_voxel_size[2])
            image.metadata["original_voxel_size"] = current_resolution
            image.scale = out_voxel_size

            # The return type still has to be Future[LayerDataTuple], otherwise other modes fails in the tests.
            result = Future()
            result.set_result(
                (
                    image.data,
                    layer_properties(
                        name=image.name,
                        scale=out_voxel_size,
                        metadata={**image.metadata},
                    ),
                    layer_type,
                )
            )
            return result

        case _:
            raise ValueError(f"{mode} is not implemented yet.")

    out_name = create_layer_name(image.name, WidgetName.RESCALING.layer_suffix)
    inputs_kwarg = {"image": image.data}
    inputs_names = (image.name,)
    step_kwargs = {"factor": rescaling_factor, "order": order}
    layer_kwargs = layer_properties(
        name=out_name, scale=out_voxel_size, metadata={**image.metadata, **{"original_voxel_size": current_resolution}}
    )

    return start_threading_process(
        image_rescale,
        runtime_kwargs=inputs_kwarg,
        statics_kwargs=step_kwargs,
        out_name=out_name,
        input_keys=inputs_names,
        layer_kwarg=layer_kwargs,
        step_name=WidgetName.RESCALING.step_name,
        layer_type=layer_type,
        viewer=viewer,
        widgets_to_update=(
            [
                widget_unet_predictions.image,
                widget_agglomeration.image,
                widget_lifted_multicut.image,
                widget_dt_ws.image,
                widget_cropping.image,
                widget_gaussian_smoothing.image,
            ]
            if update_other_widgets
            else None  # for unit tests there is no other widgets to update
        ),
    )


widget_rescaling.out_voxel_size.hide()
widget_rescaling.reference_layer.hide()
widget_rescaling.reference_model.hide()
widget_rescaling.reference_shape.hide()
widget_rescaling.reference_shape[0].max = 10000
widget_rescaling.reference_shape[1].max = 10000
widget_rescaling.reference_shape[2].max = 10000


@widget_rescaling.mode.changed.connect
def _rescale_update_visibility(mode: RescaleModes):
    mode = return_value_if_widget(mode)

    all_widgets = [
        widget_rescaling.out_voxel_size,
        widget_rescaling.reference_layer,
        widget_rescaling.reference_model,
        widget_rescaling.rescaling_factor,
        widget_rescaling.reference_shape,
    ]

    for widget in all_widgets:
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


@widget_rescaling.image.changed.connect
def _on_rescaling_image_changed(image: Layer):
    image = return_value_if_widget(image)

    if image.data.ndim == 2 or (image.data.ndim == 3 and image.data.shape[0] == 1):
        widget_rescaling.rescaling_factor[0].hide()
        widget_rescaling.reference_shape[0].hide()
        widget_rescaling.out_voxel_size[0].hide()
    else:
        widget_rescaling.rescaling_factor[0].show()
        widget_rescaling.reference_shape[0].show()
        widget_rescaling.out_voxel_size[0].show()

    for i, (shape, scale) in enumerate(zip(image.data.shape, image.scale)):
        widget_rescaling.out_voxel_size[i].value = scale
        widget_rescaling.reference_shape[i].value = shape

    if isinstance(image, Labels):
        widget_rescaling.order.value = RescaleType.NEAREST.int_val


@widget_rescaling.order.changed.connect
def _on_rescale_order_changed(order):
    order = return_value_if_widget(order)
    current_image = widget_rescaling.image.value

    if current_image is None:
        return None

    if isinstance(current_image, Labels) and order != RescaleType.NEAREST.int_val:
        napari_formatted_logging(
            "Labels can only be rescaled with nearest interpolation",
            thread=WidgetName.RESCALING.step_name,
            level="warning",
        )
        widget_rescaling.order.value = RescaleType.NEAREST.int_val


def _compute_slices(rectangle, crop_z: tuple[int, int], shape):
    z_slice = slice(*crop_z)
    if rectangle is None:
        return z_slice, slice(0, shape[1]), slice(0, shape[2])

    x_start = max(rectangle[0, 1], 0)
    x_end = min(rectangle[2, 1], shape[1])
    x_slice = slice(x_start, x_end)

    y_start = max(rectangle[0, 2], 0)
    y_end = min(rectangle[2, 2], shape[2])
    y_slice = slice(y_start, y_end)
    return z_slice, x_slice, y_slice


def _cropping(data, crop_slices):
    return data[crop_slices]


@magicgui(
    call_button=f"Run {WidgetName.CROPPING.step_name}",
    image={
        "label": "Image or Label",
        "tooltip": "Layer to apply the rescaling.",
    },
    crop_roi={
        "label": "Crop ROI",
        "tooltip": "This must be a shape layer with a rectangle XY overlaying the area to crop.",
    },
    # FloatRangeSlider and RangeSlider are not working very nicely with napari, they are usable but not very
    # nice. maybe we should use a custom widget for this.
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
    viewer: napari.Viewer,
    image: Layer,
    crop_roi: Union[Shapes, None] = None,
    crop_z: tuple[int, int] = (0, 100),
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    if crop_roi is not None:
        assert len(crop_roi.shape_type) == 1, "Only one rectangle should be used for cropping"
        assert crop_roi.shape_type[0] == "rectangle", "Only a rectangle shape should be used for cropping"

    if isinstance(image, Image):
        layer_type = "image"

    elif isinstance(image, Labels):
        layer_type = "labels"

    else:
        raise ValueError(f"{type(image)} cannot be cropped, please use Image layers or Labels layers")

    out_name = create_layer_name(image.name, WidgetName.CROPPING.layer_suffix)
    inputs_names = (image.name,)
    layer_kwargs = layer_properties(name=out_name, scale=image.scale, metadata=image.metadata)

    if crop_roi is not None:
        rectangle = crop_roi.data[0].astype("int64")
    else:
        rectangle = None

    assert isinstance(image.data, np.ndarray), "Only numpy arrays are supported for cropping."
    crop_slices = _compute_slices(rectangle, crop_z, image.data.shape)

    return start_threading_process(
        _cropping,
        runtime_kwargs={"data": image.data},
        statics_kwargs={"crop_slices": crop_slices},
        out_name=out_name,
        input_keys=inputs_names,
        layer_kwarg=layer_kwargs,
        layer_type=layer_type,
        step_name=WidgetName.CROPPING.step_name,
        skip_dag=True,
        viewer=viewer,
        widgets_to_update=(
            [
                widget_unet_predictions.image,
                widget_agglomeration.image,
                widget_lifted_multicut.image,
                widget_dt_ws.image,
                widget_rescaling.image,
                widget_gaussian_smoothing.image,
            ]
            if update_other_widgets
            else None
        ),
    )


@widget_cropping.image.changed.connect
def _on_cropping_image_changed(image: Optional[Layer]):
    if image is None:
        widget_cropping.crop_z.hide()
        return None

    image = return_value_if_widget(image)
    image_shape_z = int(image.data.shape[0])

    if image_shape_z == 1:
        widget_cropping.crop_z.hide()
        return None

    widget_cropping.crop_z.show()
    widget_cropping.crop_z.step = 1

    if widget_cropping.crop_z.value[1] > image_shape_z:
        widget_cropping.crop_z.value = (0, image_shape_z)
    widget_cropping.crop_z.max = image_shape_z


def _two_layers_operation(data1, data2, operation, weights: float = 0.5):
    if operation == "Mean":
        return weights * data1 + (1.0 - weights) * data2
    elif operation == "Maximum":
        return np.maximum(data1, data2)
    else:
        return np.minimum(data1, data2)


@magicgui(
    call_button=f"Run {WidgetName.MERGING.step_name}",
    image1={"label": "Image 1"},
    image2={"label": "Image 2"},
    operation={
        "label": "Operation",
        "tooltip": "Operation used to merge the two layers.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": ["Mean", "Maximum", "Minimum"],
    },
    weights={
        "label": "Mean weights",
        "widget_type": "FloatSlider",
        "max": 1.0,
        "min": 0.0,
    },
    update_other_widgets={
        "visible": False,
        "tooltip": "To allow toggle the update of other widgets in unit tests; invisible to users.",
    },
)
def widget_merge_layers(
    viewer: napari.Viewer,
    image1: Image,
    image2: Image,
    operation: str = "Maximum",
    weights: float = 0.5,
    update_other_widgets: bool = True,
) -> Future[LayerDataTuple]:
    out_name = create_layer_name(f"{image1.name}-{image2.name}", operation)
    inputs_names = (image1.name, image2.name)
    layer_kwargs = layer_properties(name=out_name, scale=image1.scale, metadata=image1.metadata)
    layer_type = "image"
    step_kwargs = dict(weights=weights, operation=operation)
    assert image1.data.shape == image2.data.shape

    return start_threading_process(
        _two_layers_operation,
        runtime_kwargs={"data1": image1.data, "data2": image2.data},
        statics_kwargs=step_kwargs,
        out_name=out_name,
        input_keys=inputs_names,
        layer_kwarg=layer_kwargs,
        layer_type=layer_type,
        step_name=WidgetName.MERGING.step_name,
        viewer=viewer,
        widgets_to_update=(
            [
                widget_unet_predictions.image,
                widget_agglomeration.image,
                widget_lifted_multicut.image,
                widget_dt_ws.image,
            ]
            if update_other_widgets
            else None
        ),
    )


@widget_merge_layers.operation.changed.connect
def _on_operation_changed(operation: str):
    operation = return_value_if_widget(operation)
    if operation == "Mean":
        widget_merge_layers.weights.show()
    else:
        widget_merge_layers.weights.hide()


def _label_processing(segmentation, set_bg_to_0, relabel_segmentation):
    if relabel_segmentation:
        segmentation = _relabel_segmentation(segmentation)

    if set_bg_to_0:
        segmentation = set_background_to_value(segmentation, value=0)

    return segmentation


@magicgui(
    call_button=f"Run {WidgetName.CLEANING_LABEL.step_name}",
    segmentation={
        "label": "Segmentation",
        "tooltip": "Segmentation can be any label layer.",
    },
    set_bg_to_0={
        "label": "Set background to 0",
        "tooltip": "Set the largest idx in the image to zero.",
    },
    relabel_segmentation={
        "label": "Relabel Segmentation",
        "tooltip": "Relabel segmentation contiguously to avoid labels clash.",
    },
)
def widget_label_processing(
    segmentation: Labels,
    set_bg_to_0: bool = True,
    relabel_segmentation: bool = True,
) -> Future[LayerDataTuple]:
    if relabel_segmentation and "bboxes" in segmentation.metadata.keys():
        del segmentation.metadata["bboxes"]

    out_name = create_layer_name(segmentation.name, WidgetName.CLEANING_LABEL.layer_suffix)
    inputs_kwarg = {"segmentation": segmentation.data}
    inputs_names = (segmentation.name,)
    layer_kwargs = layer_properties(name=out_name, scale=segmentation.scale, metadata=segmentation.metadata)
    layer_type = "labels"
    step_kwargs = dict(set_bg_to_0=set_bg_to_0, relabel_segmentation=relabel_segmentation)

    return start_threading_process(
        _label_processing,
        runtime_kwargs=inputs_kwarg,
        statics_kwargs=step_kwargs,
        out_name=out_name,
        input_keys=inputs_names,
        layer_kwarg=layer_kwargs,
        layer_type=layer_type,
        step_name=WidgetName.CLEANING_LABEL.step_name,
    )
