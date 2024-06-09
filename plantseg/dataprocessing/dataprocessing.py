import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import disk, ball
from vigra import gaussianSmoothing


def compute_scaling_factor(
    input_voxel_size: tuple[float, float, float], output_voxel_size: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Compute the scaling factor to rescale an image from input voxel size to output voxel size.
    """
    scaling = tuple(i_size / o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size))
    assert len(scaling) == 3, f"Expected scaling factor to be 3d, but got {len(scaling)}d input"
    return scaling


def compute_scaling_voxelsize(
    input_voxel_size: tuple[float, float, float], scaling_factor: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Compute the output voxel size after scaling an image with a given scaling factor.
    """
    output_voxel_size = tuple(i_size / s_size for i_size, s_size in zip(input_voxel_size, scaling_factor))
    assert len(output_voxel_size) == 3, f"Expected output voxel size to be 3d, but got {len(output_voxel_size)}d input"
    return output_voxel_size


def scale_image_to_voxelsize(
    image: np.ndarray,
    input_voxel_size: tuple[float, float, float],
    output_voxel_size: tuple[float, float, float],
    order: int = 0,
) -> np.ndarray:
    """
    Scale an image from a given voxel size to another voxel size.

    Args:
        image (np.ndarray): Input image to scale
        input_voxel_size (tuple[float, float, float]): Input voxel size
        output_voxel_size (tuple[float, float, float]): Output voxel size
        order (int): Interpolation order, must be 0 for segmentation and 1, 2 for images

    Returns:
        scaled_image (np.ndarray): Scaled image as numpy array
    """
    factor = compute_scaling_factor(input_voxel_size, output_voxel_size)
    return image_rescale(image, factor, order=order)


def image_rescale(image: np.ndarray, factor: tuple[float, float, float], order: int) -> np.ndarray:
    """
    Scale an image by a given factor in each dimension

    Args:
        image (np.ndarray): Input image to scale
        factor (tuple[float, float, float]): Scaling factor in each dimension
        order (int): Interpolation order, must be 0 for segmentation and 1, 2 for images

    Returns:
        scaled_image (np.ndarray): Scaled image as numpy array
    """
    if np.array_equal(factor, [1.0, 1.0, 1.0]):
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def image_median(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply median smoothing on an image with a given radius.

    Args:
        image (np.ndarray): Input image to apply median smoothing
        radius (int): Radius of the median filter

    Returns:
        median_image (np.ndarray): Median smoothed image as numpy array
    """
    if image.shape[0] == 1 or image.ndim == 2:
        shape = image.shape
        median_image = median(image[0], disk(radius))
        return median_image.reshape(shape)
    else:
        return median(image, ball(radius))


def image_gaussian_smoothing(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply gaussian smoothing on an image with a given sigma.

    Args:
        image (np.ndarray): Input image to apply gaussian smoothing
        sigma (float): Sigma value for gaussian smoothing

    Returns:
        smoothed_image (np.ndarray): Gaussian smoothed image as numpy array
    """
    image = image.astype('float32')
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma_array = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma_array)


def image_crop(image: np.ndarray, crop_str: str) -> np.ndarray:
    """
    Crop an image from a crop string like [:, 10:30:, 10:20]

    Args:
        image (np.ndarray): Input image to crop
        crop_str (str): Crop string

    Returns:
        cropped_image (np.ndarray): Cropped image as numpy array
    """
    crop_str = crop_str.replace('[', '').replace(']', '')
    slices = tuple(
        (slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip()))
        for part in crop_str.split(',')
    )
    return image[slices]


def fix_input_shape(data: np.ndarray, ndim=3) -> np.ndarray:
    assert ndim in [3, 4]
    if ndim == 3:
        return fix_input_shape_to_ZYX(data)
    else:
        return fix_input_shape_to_CZYX(data)


def fix_input_shape_to_ZYX(data: np.ndarray) -> np.ndarray:
    """
    Fix array ndim to be always 3
    """
    if data.ndim == 2:
        return data.reshape(1, data.shape[0], data.shape[1])

    elif data.ndim == 3:
        return data

    elif data.ndim == 4:
        return data[0]

    else:
        raise RuntimeError(f"Expected input data to be 2d, 3d or 4d, but got {data.ndim}d input")


def fix_input_shape_to_CZYX(data: np.ndarray) -> np.ndarray:
    """
    Fix array ndim to be 4 and return it in (C x Z x Y x X) e.g. 2 x 1 x 512 x 512
    """
    if data.ndim == 4:
        return data

    elif data.ndim == 3:
        return data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])

    else:
        raise RuntimeError(f"Expected input data to be 3d or 4d, but got {data.ndim}d input")


def normalize_01(data: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Normalize a numpy array between 0 and 1 and converts it to float32.

    Args:
        data (np.ndarray): Input numpy array
        eps (float): A small value added to the denominator for numerical stability

    Returns:
        normalized_data (np.ndarray): Normalized numpy array
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + eps).astype('float32')
