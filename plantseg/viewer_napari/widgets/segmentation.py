from typing import Optional

from magicgui import magic_factory, magicgui
from magicgui.widgets import Container, Label
from magicgui.widgets import Image as ImageW
from napari.layers import Image, Labels, Layer
from qtpy import QtGui

from plantseg import logger
from plantseg.core.image import ImageLayout, PlantSegImage, SemanticType
from plantseg.tasks.segmentation_tasks import (
    clustering_segmentation_task,
    dt_watershed_task,
    lmc_segmentation_task,
)
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.dataprocessing import (
    widget_remove_false_positives_by_foreground,
)
from plantseg.viewer_napari.widgets.prediction import Prediction_Widgets
from plantseg.viewer_napari.widgets.proofreading import (
    widget_proofreading_initialisation,
)
from plantseg.viewer_napari.widgets.utils import div, get_layers, schedule_task


class Segmentation_Tab:
    def __init__(self):
        self.STACKED = [
            ("2D Watershed", True),
            ("3D Watershed", False),
        ]
        self.AGGLOMERATION_MODES = [
            ("GASP", "gasp"),
            ("MutexWS", "mutex_ws"),
            ("MultiCut", "multicut"),
            ("LiftedMultiCut", "lmc"),
        ]
        # @@@@@ Layer selector @@@@@
        self.widget_layer_select = self.factory_layer_select()
        self.widget_layer_select.self.bind(self)
        self.widget_layer_select.layer.changed.connect(self._on_image_changed)
        self.widget_layer_select.layer._default_choices = lambda _: get_layers(
            SemanticType.PREDICTION
        )
        self.widget_layer_select.nuclei._default_choices = lambda _: get_layers(
            SemanticType.RAW
        )
        self.widget_layer_select.superpixels._default_choices = lambda _: get_layers(
            SemanticType.SEGMENTATION
        )
        font = QtGui.QFont()
        font.setBold(True)
        self.widget_layer_select.text.native.setFont(font)
        self.widget_layer_select.text.label = ""

        # @@@@@ agglomeration @@@@@
        self.widget_agglomeration = self.factory_agglomeration()
        self.widget_agglomeration.self.bind(self)
        self.widget_agglomeration.mode._default_choices = self.AGGLOMERATION_MODES
        self.widget_agglomeration.mode.reset_choices()
        self.widget_agglomeration.mode.value = self.AGGLOMERATION_MODES[0][1]
        self.widget_agglomeration.mode.changed.connect(self._on_mode_changed)

        # @@@@@ dt watershed @@@@@
        self.widget_dt_ws = self.factory_dt_ws()
        self.widget_dt_ws.self.bind(self)
        self.widget_dt_ws.stacked._default_choices = self.STACKED
        self.widget_dt_ws.stacked.reset_choices()
        self.widget_dt_ws.stacked.value = False

        self.advanced_dt_ws = [
            self.widget_dt_ws.sigma_seeds,
            self.widget_dt_ws.sigma_weights,
            self.widget_dt_ws.alpha,
            self.widget_dt_ws.use_pixel_pitch,
            self.widget_dt_ws.pixel_pitch,
            self.widget_dt_ws.apply_nonmax_suppression,
            self.widget_dt_ws.is_nuclei_image,
        ]
        for widget in self.advanced_dt_ws:
            widget.hide()
        self.widget_dt_ws.show_advanced.changed.connect(self._on_show_advanced_changed)

        self.initialised_widget_dt_ws: bool = False  # Avoid throwing an error when the first image is loaded but its layout is not supported

        self.prediction_widgets = Prediction_Widgets(self.widget_layer_select)

    def get_container(self):
        return Container(
            widgets=[
                self.widget_layer_select,
                div(),
                self.prediction_widgets.widget_unet_prediction,
                self.prediction_widgets.widget_add_custom_model,
                div(),
                self.widget_dt_ws,
                div(),
                self.widget_agglomeration,
            ],
            labels=False,
        )

    @magic_factory(
        call_button=False,
        text={
            "value": f"{'Layer selection:':<200}",
            "widget_type": "Label",
            "tooltip": (
                "Choose layers for the tasks below. Only necessary if "
                "you work with multiple images."
            ),
        },
        layer={
            "label": "Input image",
            "tooltip": "Select a layer to operate on.",
        },
        prediction={
            "label": "Boundary prediction",
            "tooltip": "Select the boundary prediction.",
        },
        nuclei={
            "label": "Nuclei image",
            "tooltip": "Nuclei foreground prediction or segmentation.",
            "visible": False,
        },
        superpixels={
            "label": "Superpixels",
            "tooltip": "Over-segmentation labels layer (superpixels) to use as input for clustering.",
        },
    )
    def factory_layer_select(
        self,
        text,
        layer: Image,
        prediction: Image,
        nuclei: Layer,
        superpixels: Labels,
    ):
        pass

    @magic_factory(
        call_button="2. Boundary to Superpixels",
        stacked={
            "label": "Mode",
            "tooltip": "Define if the Watershed will run slice by slice (faster) or on the full volume (slower).",
            "widget_type": "RadioButtons",
            "orientation": "horizontal",
        },
        threshold={
            "label": "Boundary threshold",
            "tooltip": "A low value will increase over-segmentation tendency "
            "and a large value increase under-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        min_size={
            "label": "Minimum segment size",
            "tooltip": "Minimum segment size allowed in voxels.",
        },
        # Advanced parameters
        show_advanced={
            "label": "Show advanced parameters",
            "tooltip": "Show advanced parameters for the Watershed algorithm.",
            "widget_type": "CheckBox",
        },
        sigma_seeds={"label": "Sigma seeds"},
        sigma_weights={"label": "Sigma weights"},
        alpha={"label": "Alpha"},
        use_pixel_pitch={"label": "Use pixel pitch"},
        pixel_pitch={"label": "Pixel pitch"},
        apply_nonmax_suppression={"label": "Apply nonmax suppression"},
        is_nuclei_image={"label": "Is nuclei image"},
    )
    def factory_dt_ws(
        self,
        stacked: bool,
        threshold: float = 0.5,
        min_size: int = 100,
        show_advanced: bool = False,
        sigma_seeds: float = 0.2,
        sigma_weights: float = 2.0,
        alpha: float = 1.0,
        use_pixel_pitch: bool = False,
        pixel_pitch: tuple[int, int, int] = (1, 1, 1),
        apply_nonmax_suppression: bool = False,
        is_nuclei_image: bool = False,
    ) -> None:
        ps_image = PlantSegImage.from_napari_layer(self.widget_layer_select.layer.value)

        return schedule_task(
            dt_watershed_task,
            task_kwargs={
                "image": ps_image,
                "threshold": threshold,
                "sigma_seeds": sigma_seeds,
                "stacked": stacked,
                "sigma_weights": sigma_weights,
                "min_size": min_size,
                "alpha": alpha,
                "pixel_pitch": pixel_pitch if use_pixel_pitch else None,
                "apply_nonmax_suppression": apply_nonmax_suppression,
                "is_nuclei_image": is_nuclei_image,
            },
            widgets_to_update=[],
        )

    @magic_factory(
        call_button="3.Superpixels to Segmentation",
        mode={
            "label": "Agglomeration mode",
            "tooltip": "Select which agglomeration algorithm to use.",
            "widget_type": "ComboBox",
        },
        beta={
            "label": "Under/Over segmentation factor",
            "tooltip": "A low value will increase under-segmentation tendency "
            "and a large value increase over-segmentation tendency.",
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
        },
        minsize={
            "label": "Minimum segment size",
            "tooltip": "Minimum segment size allowed in voxels.",
        },
    )
    def factory_agglomeration(
        self,
        mode: str,
        beta: float = 0.6,
        minsize: int = 100,
    ) -> None:
        ps_image = PlantSegImage.from_napari_layer(self.widget_layer_select.layer.value)
        ps_labels = PlantSegImage.from_napari_layer(
            self.widget_layer_select.superpixels.value
        )
        nuclei = self.widget_layer_select.nuclei.value

        # Hide the superpixels layer to avoid overlapping with the new segmentation
        # superpixels.visible = False

        widgets_to_update = [widget_proofreading_initialisation.segmentation]

        if mode == "lmc":
            assert isinstance(nuclei, (Image, Labels)), (
                "Nuclei must be an Image or Labels layer."
            )
            ps_nuclei = PlantSegImage.from_napari_layer(nuclei)
            return schedule_task(
                lmc_segmentation_task,
                task_kwargs={
                    "boundary_pmap": ps_image,
                    "superpixels": ps_labels,
                    "nuclei": ps_nuclei,
                    "beta": beta,
                    "post_min_size": minsize,
                },
                widgets_to_update=widgets_to_update,
            )

        return schedule_task(
            clustering_segmentation_task,
            task_kwargs={
                "image": ps_image,
                "over_segmentation": ps_labels,
                "mode": mode.lower(),
                "beta": beta,
                "post_min_size": minsize,
            },
            widgets_to_update=widgets_to_update,
        )

    def _on_mode_changed(self, mode: str):
        if mode == "lmc":
            self.widget_layer_select.nuclei.show()
        else:
            self.widget_layer_select.nuclei.hide()

    def _on_show_advanced_changed(self, state: bool):
        if state:
            self.prediction_widgets.widget_unet_prediction.advanced.value = False
            for widget in self.advanced_dt_ws:
                widget.show()
        else:
            for widget in self.advanced_dt_ws:
                widget.hide()

    def _on_image_changed(self, image: Optional[Image]):
        if image is None:
            return
        ps_image = PlantSegImage.from_napari_layer(image)

        if ps_image.image_layout == ImageLayout.ZYX:
            self.widget_dt_ws.stacked.show()
        else:
            self.widget_dt_ws.stacked.hide()
            self.widget_dt_ws.stacked.value = False
            if ps_image.image_layout != ImageLayout.YX:
                if self.initialised_widget_dt_ws:
                    log(
                        f"Unsupported image layout: {ps_image.image_layout}",
                        thread="DT Watershed",
                        level="error",
                    )
                else:
                    self.initialised_widget_dt_ws = True

    def update_layer_selection(self, layer):
        """Updates layer drop-down menus"""
        logger.debug("Updating segmentation layer drop-downs")
        print(f"UPDATE: {layer}, {layer.source}")
        self.widget_layer_select.layer.choices = get_layers(SemanticType.RAW)
        self.widget_layer_select.prediction.choices = get_layers(
            SemanticType.PREDICTION
        )
        if self.widget_layer_select.prediction.choices == ():
            self.widget_layer_select.prediction.hide()
        else:
            self.widget_layer_select.prediction.show()

        # TODO: Confirm that nuclei is indeed RAW
        self.widget_layer_select.nuclei.choices = get_layers(SemanticType.RAW)
        if self.widget_layer_select.prediction.choices == ():
            self.widget_layer_select.prediction.hide()
        else:
            self.widget_layer_select.prediction.show()

        self.widget_layer_select.superpixels.choices = get_layers(
            SemanticType.SEGMENTATION
        )
        if self.widget_layer_select.superpixels.choices == ():
            self.widget_layer_select.superpixels.hide()
        else:
            self.widget_layer_select.superpixels.show()
