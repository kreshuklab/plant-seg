from magicgui import magicgui
from napari.layers import Image, Labels, Layer

from plantseg.core.image import ImageLayout, PlantSegImage
from plantseg.tasks.segmentation_tasks import (
    clustering_segmentation_task,
    dt_watershed_task,
    lmc_segmentation_task,
)
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.dataprocessing import (
    widget_remove_false_positives_by_foreground,
)
from plantseg.viewer_napari.widgets.proofreading import (
    widget_proofreading_initialisation,
)
from plantseg.viewer_napari.widgets.utils import schedule_task

########################################################################################################################
#                                                                                                                      #
# Clustering Segmentation Widget                                                                                       #
#                                                                                                                      #
########################################################################################################################

STACKED = [
    ("2D Watershed", True),
    ("3D Watershed", False),
]
AGGLOMERATION_MODES = [
    ("GASP", "gasp"),
    ("MutexWS", "mutex_ws"),
    ("MultiCut", "multicut"),
    ("LiftedMultiCut", "lmc"),
]


@magicgui(
    call_button="Superpixels to Instance Segmentation",
    image={
        "label": "Boundary image",
        "tooltip": "Raw boundary image or boundary prediction to use as input for clustering.",
    },
    nuclei={
        "label": "Nuclei foreground",
        "tooltip": "Nuclei foreground prediction or segmentation.",
    },
    superpixels={
        "label": "Over-segmentation",
        "tooltip": "Over-segmentation labels layer (superpixels) to use as input for clustering.",
    },
    mode={
        "label": "Agglomeration mode",
        "choices": AGGLOMERATION_MODES,
        "tooltip": "Select which agglomeration algorithm to use.",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
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
def widget_agglomeration(
    image: Image,
    nuclei: Layer,
    superpixels: Labels,
    mode: str = AGGLOMERATION_MODES[0][1],
    beta: float = 0.6,
    minsize: int = 100,
) -> None:
    ps_image = PlantSegImage.from_napari_layer(image)
    ps_labels = PlantSegImage.from_napari_layer(superpixels)

    # Hide the superpixels layer to avoid overlapping with the new segmentation
    superpixels.visible = False

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


@widget_agglomeration.mode.changed.connect
def _on_mode_changed(mode: str):
    if mode == "lmc":
        widget_agglomeration.nuclei.show()
    else:
        widget_agglomeration.nuclei.hide()


########################################################################################################################
#                                                                                                                      #
# DT Watershed Segmentation Widget                                                                                     #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button="Boundary to Superpixels",
    image={
        "label": "Boundary image",
        "tooltip": "Raw boundary image or boundary prediction to use as input for Watershed.",
    },
    stacked={
        "label": "Mode",
        "tooltip": "Define if the Watershed will run slice by slice (faster) or on the full volume (slower).",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": STACKED,
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
def widget_dt_ws(
    image: Image,
    stacked: bool = False,
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
    ps_image = PlantSegImage.from_napari_layer(image)

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
        widgets_to_update=[
            widget_agglomeration.superpixels,
            widget_remove_false_positives_by_foreground.segmentation,
        ],
    )


advanced_dt_ws = [
    widget_dt_ws.sigma_seeds,
    widget_dt_ws.sigma_weights,
    widget_dt_ws.alpha,
    widget_dt_ws.use_pixel_pitch,
    widget_dt_ws.pixel_pitch,
    widget_dt_ws.apply_nonmax_suppression,
    widget_dt_ws.is_nuclei_image,
]

for widget in advanced_dt_ws:
    widget.hide()


@widget_dt_ws.show_advanced.changed.connect
def _on_show_advanced_changed(state: bool):
    if state:
        for widget in advanced_dt_ws:
            widget.show()
    else:
        for widget in advanced_dt_ws:
            widget.hide()


initialised_widget_dt_ws: bool = False  # Avoid throwing an error when the first image is loaded but its layout is not supported


@widget_dt_ws.image.changed.connect
def _on_image_changed(image: Image):
    global initialised_widget_dt_ws

    ps_image = PlantSegImage.from_napari_layer(image)

    if ps_image.image_layout == ImageLayout.ZYX:
        widget_dt_ws.stacked.show()
    else:
        widget_dt_ws.stacked.hide()
        widget_dt_ws.stacked.value = False
        if ps_image.image_layout != ImageLayout.YX:
            if initialised_widget_dt_ws:
                log(
                    f"Unsupported image layout: {ps_image.image_layout}",
                    thread="DT Watershed",
                    level="error",
                )
            else:
                initialised_widget_dt_ws = True
