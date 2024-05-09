import numpy as np
from scipy.ndimage import zoom
from skimage.filters import median
from skimage.morphology import disk, ball
from vigra import gaussianSmoothing


def compute_scaling_factor(input_voxel_size: list[float, float, float],
                           output_voxel_size: list[float, float, float]) -> list[float, float, float]:
    """
    compute the scaling factor between two voxel sizes
    """
    scaling = [i_size / o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size)]
    return scaling


def compute_scaling_voxelsize(input_voxel_size: list[float, float, float],
                              scaling_factor: list[float, float, float]) -> list[float, float, float]:
    """
    compute the output voxel size given the scaling factor
    """
    output_voxel_size = [i_size / s_size for i_size, s_size in zip(input_voxel_size, scaling_factor)]
    return output_voxel_size


def scale_image_to_voxelsize(image: np.ndarray,
                             input_voxel_size: list[float, float, float],
                             output_voxel_size: list[float, float, float],
                             order: int = 0) -> np.ndarray:
    """
    scale an image from a given voxel size
    """
    factor = compute_scaling_factor(input_voxel_size, output_voxel_size)
    return image_rescale(image, factor, order=order)


def image_rescale(image: np.ndarray, factor: list[float, float, float], order: int) -> np.ndarray:
    """
    scale an image from a given scaling factor
    """
    if np.array_equal(factor, [1., 1., 1.]):
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def image_median(image: np.ndarray, radius: int) -> np.ndarray:
    """
    apply median smoothing on an image
    """
    if image.shape[0] == 1:
        shape = image.shape
        median_image = median(image[0], disk(radius))
        return median_image.reshape(shape)
    else:
        return median(image, ball(radius))


def image_gaussian_smoothing(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    apply gaussian smoothing on an image
    """
    image = image.astype('float32')
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma_array = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma_array)


def image_crop(image: np.ndarray, crop_str: str) -> np.ndarray:
    """
    crop image from a crop string like [:, 10:30:, 10:20]
    """
    crop_str = crop_str.replace('[', '').replace(']', '')
    slices = tuple((slice(*(int(i)
                            if i else None for i in part.strip().split(':')))
                    if ':' in part else int(part.strip())) for part in crop_str.split(','))
    return image[slices]


def fix_input_shape(data: np.ndarray, ndim=3) -> np.ndarray:
    assert ndim in [3, 4]
    if ndim == 3:
        return fix_input_shape_to_ZYX(data)
    else:
        return fix_input_shape_to_CZYX(data)


def fix_input_shape_to_ZYX(data: np.ndarray) -> np.ndarray:
    """
    fix array ndim to be always 3
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


def normalize_01(data: np.ndarray) -> np.ndarray:
    """
    normalize a numpy array between 0 and 1
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-12).astype('float32')
