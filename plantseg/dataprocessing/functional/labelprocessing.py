import numpy as np
from skimage import measure


def relabel_segmentation(segmentation_image: np.ndarray) -> np.array:
    """
    Relabel contiguously a segmentation image, non-touching instances with same id will be relabeled differently.
    To be noted that measure.label is different from ndimage.label

    Args:
        segmentation_image (np.array): segmentation image to relabel
    Returns:
        np.array: relabeled segmentation image

    """
    return measure.label(segmentation_image)


def set_background_to_value(segmentation_image: np.ndarray, value: int = 0) -> np.array:
    """
    Set the largest segment (usually this is the background but not always) to a certain value

    Args:
        segmentation_image (np.array): segmentation image to relabel
        value (int): value to set the background to, default is 0
    Returns:
        np.array: relabeled segmentation image with background set to value
    """
    segmentation_image += 1
    idx, counts = np.unique(segmentation_image, return_counts=True)
    bg_idx = idx[np.argmax(counts)]
    return np.where(segmentation_image == bg_idx, value, segmentation_image)

