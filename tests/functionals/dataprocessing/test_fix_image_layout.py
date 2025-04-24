import numpy as np
import pytest

from plantseg.functionals.dataprocessing import (
    fix_layout,
)


@pytest.mark.parametrize(
    "input_shape, input_layout, output_shape, output_layout",
    [  #
        ((1, 10, 10), "ZYX", (10, 10), "YX"),
        ((10, 10, 10), "ZYX", (1, 10, 10, 10), "CZYX"),
        ((1, 10, 10, 10), "CZYX", (10, 10, 10), "ZYX"),
        ((1, 1, 10, 10), "CZYX", (10, 10), "YX"),
        ((10, 10), "YX", (1, 10, 10), "ZYX"),
        ((10, 10), "YX", (1, 1, 10, 10), "CZYX"),
        ((1, 10, 10), "CYX", (10, 10), "YX"),
        ((1, 10, 10), "CYX", (1, 10, 10), "ZYX"),
    ],
)
def test_fix_layout(input_shape, input_layout, output_shape, output_layout):
    input_image = np.random.rand(*input_shape)
    fixed_image = fix_layout(input_image, input_layout, output_layout)
    assert fixed_image.shape == output_shape
