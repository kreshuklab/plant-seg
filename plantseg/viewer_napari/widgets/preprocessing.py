from enum import Enum
from typing import Optional

from magicgui import magic_factory, magicgui
from magicgui.widgets import ComboBox, Container, EmptyWidget, Label
from napari.layers import Image, Labels, Layer, Shapes
from qtpy import QtGui

from plantseg import logger
from plantseg.core.image import ImageDimensionality, PlantSegImage
from plantseg.core.zoo import model_zoo
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks.dataprocessing_tasks import (
    ImagePairOperation,
    fix_over_under_segmentation_from_nuclei_task,
    gaussian_smoothing_task,
    image_cropping_task,
    image_pair_operation_task,
    image_rescale_to_shape_task,
    image_rescale_to_voxel_size_task,
    relabel_segmentation_task,
    remove_false_positives_by_foreground_probability_task,
    set_biggest_instance_to_zero_task,
    set_voxel_size_task,
)
from plantseg.viewer_napari import log, logger_viewer_napari
from plantseg.viewer_napari.widgets.proofreading import (
    widget_proofreading_initialisation,
)
from plantseg.viewer_napari.widgets.utils import div, schedule_task


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


class Preprocessing_Tab:
    def __init__(self):
        # @@@@@ Layer selector @@@@@
        self.widget_layer_select = self.factory_layer_select()
        self.widget_layer_select.self.bind(self)
        self.widget_layer_select.layer.changed.connect(self._on_cropping_image_changed)
        self.widget_layer_select.layer.changed.connect(self._on_layer_selection)

        font = QtGui.QFont()
        font.setBold(True)
        self.widget_layer_select.native.setFont(font)

        # @@@@@ Smoothing @@@@@
        self.widget_gaussian_smoothing = self.factory_gaussian_smoothing()
        self.widget_gaussian_smoothing.self.bind(self)

        # @@@@@ Cropping @@@@@
        self.widget_cropping = self.factory_cropping()
        self.widget_cropping.self.bind(self)

        self.initialised_widget_cropping: bool = False

        self.widget_cropping_placeholder = Container(
            widgets=[
                Label(
                    value="To crop an image, add a shape layer and draw one rectange."
                )
            ]
        )
        self.widget_cropping.hide()

        # @@@@@ Rescaling @@@@@
        self.widget_rescaling = self.factory_rescaling()
        self.widget_rescaling.self.bind(self)

        self.list_widget_rescaling_all = [
            self.widget_rescaling.out_voxel_size,
            self.widget_rescaling.reference_layer,
            self.widget_rescaling.reference_model,
            self.widget_rescaling.rescaling_factor,
            self.widget_rescaling.reference_shape,
        ]

        for widget in self.list_widget_rescaling_all:
            widget.hide()
        self.widget_rescaling.rescaling_factor.show()
        self.widget_rescaling.reference_shape[0].max = 20000
        self.widget_rescaling.reference_shape[1].max = 20000
        self.widget_rescaling.reference_shape[2].max = 20000

        self.widget_rescaling.order.changed.connect(self._on_rescale_order_changed)
        self.widget_rescaling.mode.changed.connect(self._rescale_update_visibility)

        self.list_widget_rescaling_3d = [
            self.widget_rescaling.rescaling_factor[0],
            self.widget_rescaling.reference_shape[0],
            self.widget_rescaling.out_voxel_size[0],
        ]

        # @@@@@ Image Math @@@@@
        self.widget_image_pair_operations = self.factory_image_pair_operations()
        self.widget_image_pair_operations.self.bind(self)

    def get_container(self):
        return Container(
            widgets=[
                self.widget_layer_select,
                div(),
                self.widget_cropping_placeholder,
                self.widget_cropping,
                div(),
                self.widget_rescaling,
                div(),
                self.widget_gaussian_smoothing,
                div("Image pair operations"),
                Container(
                    widgets=[EmptyWidget(label="=========\n\nImage pair operations:")]
                ),
                self.widget_image_pair_operations,
            ],
            labels=False,
        )

    @magic_factory(
        call_button=False,
        layer={
            "label": "Layer",
            "tooltip": "Select a layer to operate on.",
        },
    )
    def factory_layer_select(self, layer: Image):
        pass

    @magic_factory(
        call_button="Run Gaussian Smoothing",
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
    def factory_gaussian_smoothing(
        self, sigma: float = 1.0, update_other_widgets: bool = True
    ) -> None:
        """Apply Gaussian smoothing to an image layer."""

        ps_image = PlantSegImage.from_napari_layer(self.widget_layer_select.layer.value)

        widgets_to_update = []  # TODO

        return schedule_task(
            gaussian_smoothing_task,
            task_kwargs={
                "image": ps_image,
                "sigma": sigma,
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )

    @magic_factory(
        call_button="Crop Image",
        crop_roi={
            "label": "Crop shapes layer",
            "tooltip": "This must be a shape layer with a rectangle overlaying the area to crop.",
        },
        crop_z={
            "label": "Z slices [start, end)",
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
    def factory_cropping(
        self,
        crop_roi: Shapes | None = None,
        crop_z: tuple[int, int] = (0, 100),
        update_other_widgets: bool = True,
    ) -> None:
        layer = self.widget_layer_select.layer.value
        if crop_roi is not None:
            assert len(crop_roi.shape_type) == 1, (
                "Only one rectangle should be used for cropping"
            )
            assert crop_roi.shape_type[0] == "rectangle", (
                "Only a rectangle shape should be used for cropping"
            )

        if not isinstance(layer, (Image, Labels)):
            m = f"{type(layer)} cannot be cropped, please use Image layers or Labels layers"
            logger_viewer_napari.error(m)
            raise ValueError(m)

        if crop_roi is not None:
            rectangle = crop_roi.data[0].astype("int64")
        else:
            rectangle = None

        ps_image = PlantSegImage.from_napari_layer(layer)

        widgets_to_update = None

        return schedule_task(
            image_cropping_task,
            task_kwargs={
                "image": ps_image,
                "rectangle": rectangle,
                "crop_z": crop_z,
            },
            widgets_to_update=widgets_to_update if update_other_widgets else [],
        )

    def _on_layer_inserted_cropping(self, event):
        logger.debug("_on_layer_inserted_cropping called!")
        if isinstance(event.value, Shapes):
            logger.debug("Shapes layer added, updating cropping.")
            self.widget_cropping.crop_roi.value = event.value
            self.widget_cropping.show()
            self.widget_cropping_placeholder.hide()
        else:
            logger.debug(f"Layer added: {event.value}")

    def _on_cropping_image_changed(self, image: Optional[Layer]):
        logger.debug("_on_cropping_image_changed called!")
        if image is None:
            self.widget_cropping.crop_z.hide()
            return None

        assert isinstance(image, (Image, Labels)), (
            f"{type(image)} cannot be cropped, please use Image layers or Labels layers"
        )
        ps_image = PlantSegImage.from_napari_layer(image)

        if ps_image.dimensionality == ImageDimensionality.TWO:
            self.widget_cropping.crop_z.hide()
            return None

        if ps_image.is_multichannel:
            if self.initialised_widget_cropping:
                raise ValueError("Multichannel images are not supported for cropping.")
            else:
                self.initialised_widget_cropping = True

        self.widget_cropping.crop_z.show()
        image_shape_z = ps_image.shape[0]

        self.widget_cropping.crop_z.step = 1

        if self.widget_cropping.crop_z.value[1] > image_shape_z:
            self.widget_cropping.crop_z.value = (0, image_shape_z)
        self.widget_cropping.crop_z.max = image_shape_z

    @magic_factory(
        call_button="Rescale Image",
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
    def factory_rescaling(
        self,
        mode: RescaleModes = RescaleModes.FROM_FACTOR,
        rescaling_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        out_voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reference_layer: Layer | None = None,
        reference_model: str = model_zoo.list_models()[0],
        reference_shape: tuple[int, int, int] = (1, 1, 1),
        order: int = 0,
        update_other_widgets: bool = True,
    ) -> None:
        """Rescale an image or label layer."""

        layer = self.widget_layer_select.layer.value
        if isinstance(layer, Image) or isinstance(layer, Labels):
            ps_image = PlantSegImage.from_napari_layer(layer)
        else:
            raise ValueError("Image must be an Image or Label layer.")

        # Cover set voxel size case
        if not ps_image.has_valid_original_voxel_size():
            if mode not in [
                RescaleModes.SET_VOXEL_SIZE,
                RescaleModes.TO_LAYER_SHAPE,
                RescaleModes.TO_SHAPE,
            ]:
                raise ValueError(
                    "Original voxel size is missing, please set the voxel size manually."
                )

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
            if reference_layer is None:
                raise ValueError("Reference layer can not be None!")
            if mode == RescaleModes.TO_LAYER_SHAPE:
                output_shape = reference_layer.data.shape
            else:
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
            out_voxel_size = VoxelSize(
                voxels_size=out_voxel_size, unit=current_voxel_size.unit
            )

        if mode == RescaleModes.TO_LAYER_VOXEL_SIZE:
            if not (
                isinstance(reference_layer, Image)
                or isinstance(reference_layer, Labels)
            ):
                raise ValueError("Reference layer must be an Image or Label layer.")
            reference_ps_image = PlantSegImage.from_napari_layer(reference_layer)
            out_voxel_size = reference_ps_image.voxel_size

        if mode == RescaleModes.TO_MODEL_VOXEL_SIZE:
            model_voxel_size = model_zoo.get_model_resolution(reference_model)
            if model_voxel_size is None:
                raise ValueError(
                    f"Model {reference_model} does not have a resolution defined."
                )
            out_voxel_size = VoxelSize(
                voxels_size=model_voxel_size, unit=current_voxel_size.unit
            )

        return schedule_task(
            image_rescale_to_voxel_size_task,
            task_kwargs={
                "image": ps_image,
                "new_voxel_size": out_voxel_size,
                "order": order,
            },
            widgets_to_update=widgets_to_update,
        )

    def _rescale_update_visibility(self, mode: RescaleModes):
        logger.debug("_rescale_update_visibility called!")
        for widget in self.list_widget_rescaling_all:
            widget.hide()

        match mode:
            case RescaleModes.FROM_FACTOR:
                self.widget_rescaling.rescaling_factor.show()

            case RescaleModes.TO_LAYER_VOXEL_SIZE:
                self.widget_rescaling.reference_layer.show()

            case RescaleModes.TO_MODEL_VOXEL_SIZE:
                self.widget_rescaling.reference_model.show()

            case RescaleModes.TO_VOXEL_SIZE:
                self.widget_rescaling.out_voxel_size.show()

            case RescaleModes.TO_LAYER_SHAPE:
                self.widget_rescaling.reference_layer.show()

            case RescaleModes.TO_SHAPE:
                self.widget_rescaling.reference_shape.show()

            case RescaleModes.SET_VOXEL_SIZE:
                self.widget_rescaling.out_voxel_size.show()

            case _:
                raise ValueError(f"{mode} is not implemented yet.")

    def _on_layer_selection(self, image: Layer):
        logger.debug(f"_on_layer_selection called: {image}")
        if not (isinstance(image, Image) or isinstance(image, Labels)):
            raise ValueError("Image must be an Image or Label layer.")

        # TODO: Write tests for 2D images and labels, then change `.enabled` to `.hide()`
        if image.data.ndim == 2 or (image.data.ndim == 3 and image.data.shape[0] == 1):
            for widget in self.list_widget_rescaling_3d:
                widget.hide()
        else:
            for widget in self.list_widget_rescaling_3d:
                widget.show()

        offset = 1 if image.data.ndim == 2 else 0
        for i, (shape, scale) in enumerate(zip(image.data.shape, image.scale)):
            self.widget_rescaling.out_voxel_size[i + offset].value = scale
            self.widget_rescaling.reference_shape[i + offset].value = shape

        if isinstance(image, Labels):
            self.widget_rescaling.order.value = RescaleType.NEAREST.int_val

    def _on_rescale_order_changed(self, order):
        logger.debug("_on_rescale_order_changed called!")
        current_image = self.widget_layer_select.layer.value

        if current_image is None:
            return None

        if isinstance(current_image, Labels) and order != RescaleType.NEAREST.int_val:
            log(
                "Labels can only be rescaled with nearest interpolation",
                thread="Rescaling",
                level="warning",
            )
            self.widget_rescaling.order.value = RescaleType.NEAREST.int_val

    @magic_factory(
        call_button="Run Operation",
        layer1={
            "label": "Layer 1",
            "tooltip": "First image to apply the operation.",
        },
        layer2={
            "label": "Layer 2",
            "tooltip": "Second image to apply the operation.",
        },
        operation={
            "label": "Operation",
            "choices": ImagePairOperation,
        },
        normalize_input={
            "label": "Normalize input",
            "tooltip": "Normalize the input images to the range [0, 1].",
        },
        clip_output={
            "label": "Clip output",
            "tooltip": "Clip the output to the range [0, 1].",
        },
        normalize_output={
            "label": "Normalize output",
            "tooltip": "Normalize the output image to the range [0, 1].",
        },
    )
    def factory_image_pair_operations(
        self,
        layer1: Image,
        layer2: Image,
        operation: ImagePairOperation = "add",
        normalize_input: bool = False,
        clip_output: bool = False,
        normalize_output: bool = False,
    ) -> None:
        """Apply an operation to two image layers."""

        ps_layer1 = PlantSegImage.from_napari_layer(layer1)
        ps_layer2 = PlantSegImage.from_napari_layer(layer2)

        return schedule_task(
            image_pair_operation_task,
            task_kwargs={
                "layer1": ps_layer1,
                "layer2": ps_layer2,
                "operation": operation,
                "normalize_input": normalize_input,
                "clip_output": clip_output,
                "normalize_output": normalize_output,
            },
            widgets_to_update=[],
        )

    def update_layer_selection(self):
        """Updates layer drop-down menus"""

        def update():
            logger.debug("Updating layer names on Preprocessing tab")
            self.widget_image_pair_operations.layer1.reset_choices()
            self.widget_image_pair_operations.layer2.reset_choices()
            self.widget_rescaling.image.reset_choices()
            self.widget_cropping.image.reset_choices()
            self.widget_gaussian_smoothing.image.reset_choices()

        return update
