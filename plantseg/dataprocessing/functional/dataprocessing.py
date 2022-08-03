import numpy as np
from scipy.ndimage import zoom
from skimage.morphology import disk, ball
from skimage.filters import median
from vigra import gaussianSmoothing


def image_rescale(image, factor, order):
    if np.array_equal(factor, [1, 1, 1]):
        return image
    else:
        return zoom(image, zoom=factor, order=order)


def image_median(image, radius):
    if image.shape[0] == 1:
        shape = image.shape
        median_image = median(image[0], disk(radius))
        return median_image.reshape(shape)
    else:
        return median(image, ball(radius))


def image_gaussian_smoothing(image, sigma):
    max_sigma = (np.array(image.shape) - 1) / 3
    sigma = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
    return gaussianSmoothing(image, sigma)


def image_crop(image, crop_str):
    crop_str = crop_str.replace('[', '').replace(']', '')
    slices = tuple((slice(*(int(i)
                            if i else None for i in part.strip().split(':')))
                    if ':' in part else int(part.strip())) for part in crop_str.split(','))
    return image[slices]
