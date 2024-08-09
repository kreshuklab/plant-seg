from plantseg.functionals.segmentation import (
    dt_watershed,
    gasp,
    multicut,
    mutex_ws,
)
from plantseg.tasks import task_tracker
from plantseg.plantseg_image import PlantSegImage, SemanticType
import numpy as np
from plantseg.loggers import gui_logger


@task_tracker
def dt_watershed_task(
    image: PlantSegImage,
    threshold: float = 0.5,
    sigma_seeds: float = 1.0,
    stacked: bool = False,
    sigma_weights: float = 2.0,
    min_size: int = 100,
    alpha: float = 1.0,
    pixel_pitch: tuple[int, ...] | None = None,
    apply_nonmax_suppression: bool = False,
    n_threads: int | None = None,
    mask: np.ndarray | None = None,
) -> PlantSegImage:
    """Distance transform watershed segmentation task.

    Args:
        image (PlantSegImage): input image object
        threshold (float): threshold for the boundary probability maps
        sigma_seeds (float): sigma for the seeds
        stacked (bool): whether the boundary probability maps are stacked
        sigma_weights (float): sigma for the weights
        min_size (int): minimum size for the segments
        alpha (float): alpha parameter for the watershed
        pixel_pitch (tuple[int, ...]): pixel pitch
        apply_nonmax_suppression (bool): whether to apply non-maximum suppression
        n_threads (int): number of threads
        mask (np.ndarray): mask
    """
    if image.is_multichannel:
        raise ValueError("Multichannel images are not supported for this task.")

    if image.semantic_type != SemanticType.PREDICTION:
        gui_logger.warning(
            "The input image is not a boundary probability map. The task will still attempt to run, but the results may not be as expected."
        )

    boundary_pmaps = image.get_data()
    dt_seg = dt_watershed(
        boundary_pmaps=boundary_pmaps,
        threshold=threshold,
        sigma_seeds=sigma_seeds,
        stacked=stacked,
        sigma_weights=sigma_weights,
        min_size=min_size,
        alpha=alpha,
        pixel_pitch=pixel_pitch,
        apply_nonmax_suppression=apply_nonmax_suppression,
        n_threads=n_threads,
        mask=mask,
    )

    dt_seg_image = image.derive_new(dt_seg, name=f"{image.name}_dt_watershed", semantic_type=SemanticType.SEGMENTATION)
    return dt_seg_image


@task_tracker
def clustering_segmentation_task(
    image: PlantSegImage,
    over_segmentation: PlantSegImage | None = None,
    mode="gasp",
    beta: float = 0.5,
    post_min_size: int = 100,
) -> PlantSegImage:
    """Agglomerative segmentation task.

    Args:
        image (PlantSegImage): input image object
        over_segmentation (PlantSegImage): over-segmentation image object
        mode (str): mode for the agglomerative segmentation
        beta (float): beta parameter
        post_min_size (int): minimum size for the segments
    """
    if image.is_multichannel:
        raise ValueError("Multichannel images are not supported for this task.")

    if image.semantic_type != SemanticType.PREDICTION:
        gui_logger.warning(
            "The input image is not a boundary probability map. The task will still attempt to run, but the results may not be as expected."
        )

    if over_segmentation.semantic_type != SemanticType.SEGMENTATION:
        raise ValueError("The input over_segmentation is not a segmentation map.")

    boundary_pmaps = image.get_data()

    if over_segmentation is None:
        superpixels = None
    else:
        superpixels = over_segmentation.get_data()

        if boundary_pmaps.shape != superpixels.shape:
            raise ValueError("The boundary probability map and the over-segmentation map should have the same shape.")

    if mode == "gasp":
        seg = gasp(boundary_pmaps, superpixels=superpixels, beta=beta, post_min_size=post_min_size)
    elif mode == "multicut":
        seg = multicut(boundary_pmaps, superpixels=superpixels, beta=beta, post_min_size=post_min_size)
    elif mode == "mutex_ws":
        seg = mutex_ws(boundary_pmaps, superpixels=superpixels, beta=beta, post_min_size=post_min_size)
    else:
        raise ValueError(f"Unknown mode: {mode}, select one of ['gasp', 'multicut', 'mutex_ws']")

    seg_image = image.derive_new(seg, name=f"{image.name}_{mode}", semantic_type=SemanticType.SEGMENTATION)
    return seg_image
