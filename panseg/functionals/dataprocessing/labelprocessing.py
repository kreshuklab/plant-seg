import numpy as np
from skimage import measure  # lazy


def relabel_segmentation(
    segmentation_image: np.ndarray, background: int | None = None
) -> np.ndarray:
    r"""
    Relabels contiguously a segmentation image, non-touching instances with same id will be relabeled differently.
    To be noted that measure.label is different from ndimage.label.

    1-connectivity     2-connectivity     diagonal connection close-up

         [ ]           [ ]  [ ]  [ ]             [ ]
          |               \  |  /                 |  <- hop 2
    [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
          |               /  |  \             hop 1
         [ ]           [ ]  [ ]  [ ]

    Args:
        segmentation_image (np.ndarray): A 2D or 3D segmentation image where connected components represent different instances.
        background (int | None, optional): Label of the background. If None, the function will assume the background
                                           label is 0. Default is None.

    Returns:
        np.ndarray: A relabeled segmentation image where each connected component is assigned a unique integer label.
    """
    relabeled_segmentation = measure.label(
        segmentation_image,
        background=background,
        return_num=False,
        connectivity=None,
    )
    assert isinstance(relabeled_segmentation, np.ndarray)
    return relabeled_segmentation


def get_largest_instance_id(
    segmentation: np.ndarray, include_zero: bool = False
) -> int:
    """
    Returns the label of the largest instance in the segmentation image based on pixel count.

    Args:
        segmentation (np.ndarray): A 2D or 3D segmentation image.
        include_zero (bool, optional): Whether to include the background (label 0) in the computation.
                                       Default is False.

    Returns:
        int: The label of the largest instance in the segmentation image.
    """
    instance_ids, counts = np.unique(segmentation, return_counts=True)

    if not include_zero and 0 in instance_ids:
        instance_ids = instance_ids[1:]
        counts = counts[1:]

    largest_instance_index = np.argmax(counts)
    largest_instance_id = instance_ids[largest_instance_index]

    return largest_instance_id


def set_biggest_instance_to_value(
    segmentation_image: np.ndarray, value: int = 0, instance_could_be_zero: bool = False
) -> np.ndarray:
    """
    Sets the largest segment in the segmentation image to a specified value.

    This function identifies the largest connected component (by pixel count) in the segmentation image and
    replaces its label with the specified value. Note that if the new value already exists in the image, it
    could lead to ambiguous labels.

    Args:
        segmentation_image (np.ndarray): A 2D or 3D numpy array representing an instance segmentation.
        value (int, optional): The value to assign to the largest segment. Default is 0.
        instance_could_be_zero (bool, optional): If True, treats label 0 as a valid instance label rather than background.
                                                 In this case, 1 is added to all labels before processing. Default is False.

    Returns:
        np.ndarray: The segmentation image with the largest instance set to `value`.
    """
    largest_label = get_largest_instance_id(
        segmentation_image, include_zero=instance_could_be_zero
    )
    modified_segmentation_image = np.where(
        segmentation_image == largest_label, value, segmentation_image
    )

    return modified_segmentation_image


def set_biggest_instance_to_zero(
    segmentation_image: np.ndarray, instance_could_be_zero: bool = False
) -> np.ndarray:
    """
    Sets the largest segment in the segmentation image to zero.

    This function identifies the largest connected component (by pixel count) in the segmentation image and
    replaces its label with zero. By default, label 0 is considered the background, but this can be altered
    with the `instance_could_be_zero` parameter.

    Args:
        segmentation_image (np.ndarray): A 2D or 3D numpy array representing an instance segmentation.
        instance_could_be_zero (bool, optional): If True, treats label 0 as a valid instance label. Default is False.

    Returns:
        np.ndarray: The segmentation image with the largest instance set to 0.
    """
    return set_biggest_instance_to_value(
        segmentation_image, value=0, instance_could_be_zero=instance_could_be_zero
    )


def set_value_to_value(
    segmentation_image: np.ndarray, value: int = 0, new_value: int = 0
) -> np.ndarray:
    """
    Replaces all occurrences of a specific value in the segmentation image with a new value.

    Args:
        segmentation_image (np.ndarray): A 2D or 3D numpy array representing an instance segmentation.
        value (int, optional): The value to be replaced. Default is 0.
        new_value (int, optional): The new value to assign in place of the specified value. Default is 0.

    Returns:
        np.ndarray: A segmentation image where all occurrences of `value` have been replaced with `new_value`.
    """
    return np.where(segmentation_image == value, new_value, segmentation_image)


def set_background_to_value(
    segmentation_image: np.ndarray, value: int = 0
) -> np.ndarray:
    """
    Sets all occurrences of the background (label 0) in the segmentation image to a new value.

    Args:
        segmentation_image (np.ndarray): A 2D or 3D numpy array representing an instance segmentation.
        value (int, optional): The value to assign to the background. Default is 0.

    Returns:
        np.ndarray: A segmentation image where all background pixels (originally 0) are set to `value`.
    """
    return np.where(segmentation_image == 0, value, segmentation_image)
