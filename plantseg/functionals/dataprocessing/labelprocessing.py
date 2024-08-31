import numpy as np
from skimage import measure


def relabel_segmentation(segmentation_image: np.ndarray) -> np.ndarray:
    r"""
    Relabel contiguously a segmentation image, non-touching instances with same id will be relabeled differently.
    To be noted that measure.label is different from ndimage.label.

    1-connectivity     2-connectivity     diagonal connection close-up

         [ ]           [ ]  [ ]  [ ]             [ ]
          |               \  |  /                 |  <- hop 2
    [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
          |               /  |  \             hop 1
         [ ]           [ ]  [ ]  [ ]

    Args:
        segmentation_image (np.ndarray): segmentation image to relabel

    Returns:
        new_segmentation_image (np.ndarray): relabeled segmentation image

    """
    return measure.label(
        segmentation_image,
        background=0,
        return_num=False,
        connectivity=2,
    )


def set_biggest_instance_to_value(
    segmentation_image: np.ndarray, value: int = 0, instance_could_be_zero=False
) -> np.ndarray:
    """
    Set the largest segment (usually this is the background but not always) to a certain value.

    This could cause problem if the new value already exists in the segmentation image.

    Args:
        segmentation_image (np.ndarray): segmentation image to relabel
        value (int): value to set the background to, default is 0
        instance_could_be_zero (bool): if True, 0 might be an instance label, add 1 to all labels before processing

    Returns:
        new_segmentation_image (np.ndarray): segmentation image with background set to value
    """
    if instance_could_be_zero:
        segmentation_image += 1
    idx, counts = np.unique(segmentation_image, return_counts=True)
    bg_idx = idx[np.argmax(counts)]
    return np.where(segmentation_image == bg_idx, value, segmentation_image)


def set_biggest_instance_to_zero(segmentation_image: np.ndarray) -> np.ndarray:
    """
    Set the largest segment (usually this is the background but not always) to zero.

    Args:
        segmentation_image (np.ndarray): segmentation image to relabel

    Returns:
        new_segmentation_image (np.ndarray): segmentation image with background set to 0
    """
    return set_biggest_instance_to_value(segmentation_image, 0)


def set_value_to_value(segmentation_image: np.ndarray, value: int = 0, new_value: int = 0) -> np.ndarray:
    """
    Set the specified value to a certain value.

    Args:
        segmentation_image (np.ndarray): segmentation image to relabel
        value (int): value to change, default is 0
        new_value (int): value to set the specified value to, default is 0

    Returns:
        new_segmentation_image (np.ndarray): segmentation image with specified value set to new_value
    """
    return np.where(segmentation_image == value, new_value, segmentation_image)


def set_background_to_value(segmentation_image: np.ndarray, value: int = 0) -> np.ndarray:
    """
    Set 0s (usually this is the background but not always) to a certain value.

    Args:
        segmentation_image (np.ndarray): segmentation image to relabel
        value (int): value to set the background to, default is 0

    Returns:
        new_segmentation_image (np.ndarray): segmentation image with background set to value
    """
    return np.where(segmentation_image == 0, value, segmentation_image)
