import numpy as np
import pytest

from plantseg.functionals.dataprocessing import (
    add_image,
    divide_image,
    max_image,
    multiply_image,
    process_images,
    subtract_image,
)


# Mock normalize_01 function (replace with your actual implementation)
def normalize_01(image: np.ndarray) -> np.ndarray:
    min_val, max_val = image.min(), image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


# Test cases for process_images
@pytest.mark.parametrize(
    "operation, expected_result",
    [
        ("add", np.array([[2, 4], [6, 8]])),
        ("multiply", np.array([[1, 4], [9, 16]])),
        ("subtract", np.array([[0, 0], [0, 0]])),
        ("divide", np.array([[1, 1], [1, 1]])),
        ("max", np.array([[1, 2], [3, 4]])),
    ],
)
def test_process_images(operation, expected_result):
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = process_images(
        image1, image2, operation=operation, normalize_in=False, clipping=False, normalize_out=False
    )
    assert np.allclose(result, expected_result), f"Failed for operation: {operation}"


def test_process_images_clipping():
    image1 = np.array([[0.5, 1.5], [2.5, 3.5]])
    image2 = np.array([[1.0, 1.0], [1.0, 1.0]])

    result = process_images(image1, image2, operation="add", clipping=True, normalize_out=False)
    expected = np.clip(image1 + image2, 0, 1)
    assert np.allclose(result, expected), "Clipping failed"


def test_process_images_normalize_in_out():
    image1 = np.array([[10, 20], [30, 40]])
    image2 = np.array([[1, 2], [3, 4]])

    result = process_images(image1, image2, operation="add", normalize_in=True, normalize_out=True)
    assert np.allclose(result, normalize_01(result)), "Normalization failed"


# Test cases for specific operations
def test_add_image():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = add_image(image1, image2, clipping=False, normalize=False, preprocess=False)
    expected = image1 + image2
    assert np.allclose(result, expected), "Addition failed"


def test_multiply_image():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = multiply_image(image1, image2, clipping=False, normalize=False, preprocess=False)
    expected = image1 * image2
    assert np.allclose(result, expected), "Multiplication failed"


def test_subtract_image():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = subtract_image(image1, image2, clipping=False, normalize=False, preprocess=False)
    expected = image1 - image2
    assert np.allclose(result, expected), "Subtraction failed"


def test_divide_image():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    result = divide_image(image1, image2, clipping=False, normalize=False, preprocess=False)
    expected = image1 / image2
    assert np.allclose(result, expected), "Division failed"


def test_max_image():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[4, 3], [2, 1]])

    result = max_image(image1, image2, clipping=False, normalize=False, preprocess=False)
    expected = np.maximum(image1, image2)
    assert np.allclose(result, expected), "Max operation failed"


# Error handling
def test_process_images_invalid_operation():
    image1 = np.array([[1, 2], [3, 4]])
    image2 = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Unsupported operation: invalid"):
        process_images(image1, image2, operation="invalid", normalize_in=False, clipping=False, normalize_out=False)
