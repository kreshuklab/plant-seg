from enum import Enum
from typing import Optional

from magicgui import magic_factory, magicgui
from magicgui.widgets import Container
from napari.layers import Image, Labels, Layer, Shapes

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
from plantseg.viewer_napari import log
from plantseg.viewer_napari.widgets.proofreading import (
    widget_proofreading_initialisation,
)
from plantseg.viewer_napari.widgets.utils import schedule_task


class Postprocessing_Tab:
    def __init__(self):
        # @@@@@ Relable @@@@@
        self.widget_relabel = self.factory_relabel()
        self.widget_relabel.self.bind(self)
        self.widget_relabel.segmentation.changed.connect(
            self._on_relabel_segmentation_changed
        )

        # @@@@@ Set biggest instance to zero @@@@@
        self.widget_set_biggest_instance_zero = self.factory_set_biggest_instance_zero()
        self.widget_set_biggest_instance_zero.self.bind(self)

        # @@@@@ remove false positive @@@@@
        self.widget_remove_false_positives_by_foreground = (
            self.factory_remove_false_positives_by_foreground()
        )
        self.widget_remove_false_positives_by_foreground.self.bind(self)

        # @@@@@ fix segmentation by nuclei @@@@@
        self.widget_fix_over_under_segmentation_from_nuclei = (
            self.factory_fix_over_under_segmentation_from_nuclei()
        )
        self.widget_fix_over_under_segmentation_from_nuclei.self.bind(self)

    def get_container(self):
        return Container(
            widgets=[
                self.widget_relabel,
                self.widget_set_biggest_instance_zero,
                self.widget_remove_false_positives_by_foreground,
                self.widget_fix_over_under_segmentation_from_nuclei,
            ],
            labels=False,
        )

    @magic_factory(
        call_button="Relabel Instances",
        segmentation={
            "label": "Segmentation",
            "tooltip": "Segmentation can be any label layer.",
        },
        background={
            "label": "Background label",
            "tooltip": "Background label will be set to 0. Default is None.",
            "max": 1000,
            "min": 0,
        },
    )
    def factory_relabel(
        self,
        segmentation: Labels,
        background: int | None = None,
    ) -> None:
        """Relabel an image layer."""

        ps_image = PlantSegImage.from_napari_layer(segmentation)

        segmentation.visible = False
        widgets_to_update = [
            # widget_relabel.segmentation,
            # widget_set_biggest_instance_to_zero.segmentation,
            # widget_remove_false_positives_by_foreground.segmentation,
            # widget_fix_over_under_segmentation_from_nuclei.segmentation_cells,
            # widget_proofreading_initialisation.segmentation,
        ]
        return schedule_task(
            relabel_segmentation_task,
            task_kwargs={
                "image": ps_image,
                "background": background,
            },
            widgets_to_update=widgets_to_update,
        )

    def _on_relabel_segmentation_changed(self, segmentation: Optional[Labels]):
        if segmentation is None:
            self.widget_relabel.background.hide()
            return None

        self.widget_relabel.background.max = int(segmentation.data.max())

    @magic_factory(
        call_button="Set Biggest Instance to Zero",
        segmentation={
            "label": "Segmentation",
            "tooltip": "Segmentation can be any label layer.",
        },
        instance_could_be_zero={
            "label": "Treat 0 as instance",
            "tooltip": "If ticked, a proper instance segmentation with 0 as background will not be modified.",
        },
    )
    def factory_set_biggest_instance_zero(
        self,
        segmentation: Labels,
        instance_could_be_zero: bool = False,
    ) -> None:
        """Set the biggest instance to zero in a label layer."""

        ps_image = PlantSegImage.from_napari_layer(segmentation)

        segmentation.visible = False
        widgets_to_update = [
            # widget_relabel.segmentation,
            # widget_set_biggest_instance_to_zero.segmentation,
            # widget_remove_false_positives_by_foreground.segmentation,
            # widget_fix_over_under_segmentation_from_nuclei.segmentation_cells,
            # widget_proofreading_initialisation.segmentation,
        ]
        return schedule_task(
            set_biggest_instance_to_zero_task,
            task_kwargs={
                "image": ps_image,
                "instance_could_be_zero": instance_could_be_zero,
            },
            widgets_to_update=widgets_to_update,
        )

    @magic_factory(
        call_button="Remove Objects with Low Foreground Probability",
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
            "widget_type": "FloatSlider",
            "max": 1.0,
            "min": 0.0,
            "step": 0.01,
        },
    )
    def factory_remove_false_positives_by_foreground(
        self, segmentation: Labels, foreground: Image, threshold: float = 0.5
    ) -> None:
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

    @magic_factory(
        call_button="Split/Merge Instances by Nuclei",
        segmentation_cells={"label": "Cell instances"},
        segmentation_nuclei={"label": "Nuclear instances"},
        boundary_pmaps={"label": "Boundary image"},
        threshold={
            "label": "Boundary Threshold (%)",
            "tooltip": "Set the percentage range for merging (first value) and splitting (second value) cells. "
            'For example, "33" means cells with less than 33% overlap with nuclei are merged, and '
            '"66" means cells with more than 66% overlap are split.',
            "widget_type": "FloatRangeSlider",
            "max": 100,
            "min": 0,
            "step": 0.1,
        },
        quantile={
            "label": "Nuclei Size Filter (%)",
            "tooltip": "Set the size range to filter nuclei, represented as percentages. "
            'For example, "0.3" excludes the smallest 30%, and "99.9" excludes the largest 0.1% of nuclei.',
            "widget_type": "FloatRangeSlider",
            "max": 100,
            "min": 0,
            "step": 0.1,
        },
    )
    def factory_fix_over_under_segmentation_from_nuclei(
        self,
        segmentation_cells: Labels,
        segmentation_nuclei: Labels,
        boundary_pmaps: Image | None = None,
        threshold=(33, 66),
        quantile=(0.3, 99.9),
    ) -> None:
        """
        Widget for correcting over- and under-segmentation of cells based on nuclei segmentation.

        This widget allows users to adjust cell segmentation by leveraging nuclei segmentation. It supports
        merging over-segmented cells and splitting under-segmented cells, with optional boundary refinement.

        Args:
            segmentation_cells (Labels): Input layer representing segmented cell instances.
            segmentation_nuclei (Labels): Input layer representing segmented nuclei instances.
            boundary_pmaps (Image | None, optional): Optional boundary probability map (same shape as input layers).
                Higher values indicate probable cell boundaries, used to refine segmentation.
            threshold (tuple[float, float], optional): Merge and split thresholds as percentages (0-100).
                - The first value is the merge threshold: cells with nuclei overlap below this value are merged.
                - The second value is the split threshold: cells with nuclei overlap above this value are split.
                Default is (33, 66).
            quantile (tuple[float, float], optional): Minimum and maximum quantile values for filtering nuclei sizes (0-100).
                - The first value excludes the smallest nuclei (e.g., "0.3" excludes the smallest 0.3%).
                - The second value excludes the largest nuclei (e.g., "99.9" excludes the largest 0.1%).
                Default is (0.3, 99.9).

        Returns:
            None
        """
        ps_seg_cel = PlantSegImage.from_napari_layer(segmentation_cells)
        ps_seg_nuc = PlantSegImage.from_napari_layer(segmentation_nuclei)
        ps_pmap_cell_boundary = (
            PlantSegImage.from_napari_layer(boundary_pmaps) if boundary_pmaps else None
        )

        # Normalize percentages to fractions
        threshold_merge = threshold[0] / 100
        threshold_split = threshold[1] / 100
        quantile_min = quantile[0] / 100
        quantile_max = quantile[1] / 100

        return schedule_task(
            fix_over_under_segmentation_from_nuclei_task,
            task_kwargs={
                "cell_seg": ps_seg_cel,
                "nuclei_seg": ps_seg_nuc,
                "threshold_merge": threshold_merge,
                "threshold_split": threshold_split,
                "quantile_min": quantile_min,
                "quantile_max": quantile_max,
                "boundary": ps_pmap_cell_boundary,
            },
            widgets_to_update=[],
        )
