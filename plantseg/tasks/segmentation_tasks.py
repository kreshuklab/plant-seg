import logging

from plantseg.core.image import ImageLayout, PlantSegImage, SemanticType
from plantseg.functionals.dataprocessing.dataprocessing import normalize_01
from plantseg.functionals.segmentation import (
    dt_watershed,
    gasp,
    lifted_multicut_from_nuclei_pmaps,
    lifted_multicut_from_nuclei_segmentation,
    multicut,
    mutex_ws,
)
from plantseg.tasks import task_tracker

logger = logging.getLogger(__name__)


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
    is_nuclei_image: bool = False,
) -> PlantSegImage:
    """Distance transform watershed segmentation task.

    This function applies the distance transform watershed algorithm to segment the input image.
    It handles both standard boundary probability maps and nuclei images, with options for
    various preprocessing and segmentation parameters.

    Args:
        image (PlantSegImage): The input image to segment.
        threshold (float, optional): Threshold value for the boundary probability maps.
            Defaults to 0.5.
        sigma_seeds (float, optional): Standard deviation for Gaussian smoothing applied to
            the seed map. Defaults to 1.0.
        stacked (bool, optional): If True and the image is 3D, processes the image
            slice-by-slice (2D). Defaults to False.
        sigma_weights (float, optional): Standard deviation for Gaussian smoothing applied to
            the weight map. Defaults to 2.0.
        min_size (int, optional): Minimum size of the segments to keep. Smaller segments
            will be removed. Defaults to 100.
        alpha (float, optional): Blending factor between the input image and the distance
            transform when computing the weight map. Defaults to 1.0.
        pixel_pitch (tuple[int, ...] | None, optional): Anisotropy factors for the distance
            transform. If None, isotropic distances are assumed. Defaults to None.
        apply_nonmax_suppression (bool, optional): Whether to apply non-maximum suppression
            to the seeds. Requires the Nifty library. Defaults to False.
        n_threads (int | None, optional): Number of threads to use for parallel processing
            in 2D mode. Defaults to None.
        is_nuclei_image (bool, optional): If True, indicates that the input image is a nuclei
            image, and preprocessing is applied accordingly. Defaults to False.

    Returns:
        PlantSegImage: The segmented image as a new `PlantSegImage` object.
    """
    if image.is_multichannel:
        raise ValueError("Multichannel images are not supported for this task.")

    if image.semantic_type != SemanticType.PREDICTION:
        logger.warning(
            "The input image is not a boundary probability map. The task will still attempt to run, but the results may not be as expected."
        )

    if image.image_layout == ImageLayout.YX and stacked:
        logger.warning(
            "Stack, or 'per slice' is only for 3D images (ZYX). The stack option will be disabled."
        )
        stacked = False

    if is_nuclei_image:
        boundary_pmaps = normalize_01(image.get_data())
        boundary_pmaps = 1.0 - boundary_pmaps
        mask = boundary_pmaps < threshold
    else:
        boundary_pmaps = image.get_data()
        mask = None

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

    dt_seg_image = image.derive_new(
        dt_seg,
        name=f"{image.name}_dt_watershed",
        semantic_type=SemanticType.SEGMENTATION,
    )
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
        logger.warning(
            "The input image is not a boundary probability map. The task will still attempt to run, but the results may not be as expected."
        )

    boundary_pmaps = image.get_data()

    if over_segmentation is None:
        superpixels = None
    else:
        if over_segmentation.semantic_type != SemanticType.SEGMENTATION:
            raise ValueError("The input over_segmentation is not a segmentation map.")
        superpixels = over_segmentation.get_data()

        if boundary_pmaps.shape != superpixels.shape:
            raise ValueError(
                "The boundary probability map and the over-segmentation map should have the same shape."
            )

    if mode == "gasp":
        seg = gasp(
            boundary_pmaps,
            superpixels=superpixels,
            beta=beta,
            post_minsize=post_min_size,
        )
    elif mode == "multicut":
        if superpixels is None:
            raise ValueError("The superpixels are required for the multicut mode.")
        seg = multicut(
            boundary_pmaps,
            superpixels=superpixels,
            beta=beta,
            post_minsize=post_min_size,
        )
    elif mode == "mutex_ws":
        seg = mutex_ws(
            boundary_pmaps,
            superpixels=superpixels,
            beta=beta,
            post_minsize=post_min_size,
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode}, select one of ['gasp', 'multicut', 'mutex_ws']"
        )

    seg_image = image.derive_new(
        seg, name=f"{image.name}_{mode}", semantic_type=SemanticType.SEGMENTATION
    )
    return seg_image


@task_tracker
def lmc_segmentation_task(
    boundary_pmap: PlantSegImage,
    superpixels: PlantSegImage,
    nuclei: PlantSegImage,
    beta: float = 0.5,
    post_min_size: int = 100,
) -> PlantSegImage:
    """Lifted multicut segmentation task.

    Args:
        boundary_pmap (PlantSegImage): cell boundary prediction, PlantSegImage of shape (Z, Y, X) with values between 0 and 1.
        superpixels (PlantSegImage): superpixels/over-segmentation. Must have the same shape as boundary_pmap.
        nuclei (PlantSegImage): a nuclear segmentation or prediction map. Must have the same shape as boundary_pmap.
        beta (float): beta parameter for the Multicut.
            A small value will steer the segmentation towards under-segmentation, while
            a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_min_size (int): minimal size of the segments after Multicut. (default: 100)
    """
    if (
        nuclei.semantic_type is SemanticType.PREDICTION
        or nuclei.semantic_type is SemanticType.RAW
    ):
        lmc = lifted_multicut_from_nuclei_pmaps
        extra_key = "nuclei_pmaps"
    else:
        lmc = lifted_multicut_from_nuclei_segmentation
        extra_key = "nuclei_seg"

    segmentation = lmc(
        boundary_pmaps=boundary_pmap.get_data(),
        superpixels=superpixels.get_data(),
        **{extra_key: nuclei.get_data()},
        beta=beta,
        post_minsize=post_min_size,
    )

    ps_seg = superpixels.derive_new(
        segmentation,
        name=f"{superpixels.name}_lmc",
        semantic_type=SemanticType.SEGMENTATION,
    )
    return ps_seg
