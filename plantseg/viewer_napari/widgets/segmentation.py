from concurrent.futures import Future

from magicgui import magicgui
from napari.layers import Image, Labels
from napari.types import LayerDataTuple

from plantseg.plantseg_image import PlantSegImage
from plantseg.tasks.segmentation_tasks import clustering_segmentation_task, dt_watershed_task, lmc_segmentation_task
from plantseg.viewer_napari.widgets.utils import schedule_task

########################################################################################################################
#                                                                                                                      #
# Clustering Segmentation Widget                                                                                       #
#                                                                                                                      #
########################################################################################################################

STACKED = [('2D', True), ('3D', False)]


@magicgui(
    call_button='Run Clustering',
    image={
        'label': 'Pmap/Image',
        'tooltip': 'Raw or boundary image to use as input for clustering.',
    },
    _labels={
        'label': 'Over-segmentation',
        'tooltip': 'Over-segmentation labels layer to use as input for clustering.',
    },
    mode={
        'label': 'Aggl. Mode',
        'choices': ['GASP', 'MutexWS', 'MultiCut'],
        'tooltip': 'Select which agglomeration algorithm to use.',
    },
    beta={
        'label': 'Under/Over segmentation factor',
        'tooltip': 'A low value will increase under-segmentation tendency '
        'and a large value increase over-segmentation tendency.',
        'widget_type': 'FloatSlider',
        'max': 1.0,
        'min': 0.0,
    },
    minsize={
        'label': 'Min-size',
        'tooltip': 'Minimum segment size allowed in voxels.',
    },
)
def widget_agglomeration(
    image: Image,
    _labels: Labels,
    mode: str = "GASP",
    beta: float = 0.6,
    minsize: int = 100,
) -> Future[LayerDataTuple]:
    ps_image = PlantSegImage.from_napari_layer(image)
    ps_labels = PlantSegImage.from_napari_layer(_labels)

    return schedule_task(
        clustering_segmentation_task,
        task_kwargs={
            "image": ps_image,
            "over_segmentation": ps_labels,
            "mode": mode.lower(),
            "beta": beta,
            "post_min_size": minsize,
        },
    )


########################################################################################################################
#                                                                                                                      #
# Lifted Multicut Segmentation Widget                                                                                  #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button='Run Lifted MultiCut',
    image={
        'label': 'Pmap/Image',
        'tooltip': 'Raw or boundary image to use as input for clustering.',
    },
    nuclei={
        'label': 'Nuclei',
        'tooltip': 'Nuclei binary predictions or Nuclei segmentation.',
    },
    _labels={
        'label': 'Over-segmentation',
        'tooltip': 'Over-segmentation labels layer to use as input for clustering.',
    },
    beta={
        'label': 'Under/Over segmentation factor',
        'tooltip': 'A low value will increase under-segmentation tendency '
        'and a large value increase over-segmentation tendency.',
        'widget_type': 'FloatSlider',
        'max': 1.0,
        'min': 0.0,
    },
    minsize={
        'label': 'Min-size',
        'tooltip': 'Minimum segment size allowed in voxels.',
    },
)
def widget_lifted_multicut(
    image: Image,
    nuclei: Image | Labels,
    _labels: Labels,
    beta: float = 0.5,
    minsize: int = 100,
) -> Future[LayerDataTuple]:
    ps_image = PlantSegImage.from_napari_layer(image)
    ps_labels = PlantSegImage.from_napari_layer(_labels)
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
    )


########################################################################################################################
#                                                                                                                      #
# DT Watershed Segmentation Widget                                                                                     #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button='Run Watershed',
    image={
        'label': 'Image or Probability Map',
        'tooltip': 'Raw or boundary image to use as input for Watershed.',
    },
    stacked={
        'label': 'Stacked',
        'tooltip': 'Define if the Watershed will run slice by slice (faster) ' 'or on the full volume (slower).',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': STACKED,
    },
    threshold={
        'label': 'Threshold',
        'tooltip': 'A low value will increase over-segmentation tendency '
        'and a large value increase under-segmentation tendency.',
        'widget_type': 'FloatSlider',
        'max': 1.0,
        'min': 0.0,
    },
    min_size={
        'label': 'Minimum segment size',
        'tooltip': 'Minimum segment size allowed in voxels.',
    },
    # Advanced parameters
    show_advanced={
        'label': 'Show Advanced Parameters',
        'tooltip': 'Show advanced parameters for the Watershed algorithm.',
        'widget_type': 'CheckBox',
    },
    sigma_seeds={'label': 'Sigma seeds'},
    sigma_weights={'label': 'Sigma weights'},
    alpha={'label': 'Alpha'},
    use_pixel_pitch={'label': 'Use pixel pitch'},
    pixel_pitch={'label': 'Pixel pitch'},
    apply_nonmax_suppression={'label': 'Apply nonmax suppression'},
    is_nuclei_image={'label': 'Is nuclei image'},
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
) -> Future[LayerDataTuple]:
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
    )


widget_dt_ws.sigma_seeds.hide()
widget_dt_ws.sigma_weights.hide()
widget_dt_ws.alpha.hide()
widget_dt_ws.use_pixel_pitch.hide()
widget_dt_ws.pixel_pitch.hide()
widget_dt_ws.apply_nonmax_suppression.hide()
widget_dt_ws.is_nuclei_image.hide()


@widget_dt_ws.show_advanced.changed.connect
def _on_show_advanced_changed(state: bool):
    if state:
        widget_dt_ws.sigma_seeds.show()
        widget_dt_ws.sigma_weights.show()
        widget_dt_ws.alpha.show()
        widget_dt_ws.use_pixel_pitch.show()
        widget_dt_ws.pixel_pitch.show()
        widget_dt_ws.apply_nonmax_suppression.show()
        widget_dt_ws.is_nuclei_image.show()
    else:
        widget_dt_ws.sigma_seeds.hide()
        widget_dt_ws.sigma_weights.hide()
        widget_dt_ws.alpha.hide()
        widget_dt_ws.use_pixel_pitch.hide()
        widget_dt_ws.pixel_pitch.hide()
        widget_dt_ws.apply_nonmax_suppression.hide()
        widget_dt_ws.is_nuclei_image.hide()
