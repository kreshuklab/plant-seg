import pytest

from plantseg.functionals.predictions.utils.array_predictor import find_patch_and_halo_shapes


@pytest.mark.parametrize(
    "full_volume_shape, max_patch_shape, min_halo_shape, expected",
    [
        ((1, 12000, 50000), (192, 192, 192), (88, 88, 88), ((1, 2572, 2572), (0, 88, 88))),
        ((1, 120, 500), (192, 192, 192), (88, 88, 88), ((1, 120, 500), (0, 0, 0))),
        ((95, 120, 500), (192, 192, 192), (88, 88, 88), ((95, 120, 500), (0, 0, 0))),
        ((95, 120, 5000), (192, 192, 192), (88, 88, 88), ((95, 120, 532), (0, 0, 88))),
        ((5000, 120, 95), (192, 192, 192), (88, 88, 88), ((532, 120, 95), (88, 0, 0))),
        ((100, 1000, 1000), (192, 192, 192), (88, 88, 88), ((100, 178, 178), (0, 88, 88))),
        ((1000, 1000, 1000), (192, 192, 192), (88, 88, 88), ((104, 104, 104), (88, 88, 88))),
    ],
)
def test_find_patch_and_halo_shapes(full_volume_shape, max_patch_shape, min_halo_shape, expected):
    result = find_patch_and_halo_shapes(full_volume_shape, max_patch_shape, min_halo_shape)
    assert result == expected
