import numpy as np
import pytest
from pydantic import ValidationError

from plantseg.io.voxelsize import VoxelSize


@pytest.mark.parametrize(
    "unit",
    [
        "",
        "-",
        "um",
        "μm",
        "\u03bcm",
        r"\u03bcm",
        "\u00b5m",
        r"\u00b5m",
        "micrometer",
    ],
)
def test_voxel_size_initialization(unit):
    """Test valid initialization with different units."""
    voxel_size = (0.5, 0.5, 0.5)
    vs = VoxelSize(voxels_size=voxel_size, unit=unit)
    assert vs.voxels_size == voxel_size
    assert vs.unit == "um"


@pytest.mark.parametrize(
    "unit",
    [
        "ums",
        "μ",
        "\u03bcmm",
        r"\u03bcm2",
        "\u00b5M",
        r"\u00b5s",
        "micrometers",
    ],
)
def test_voxel_size_crazy_initialization(unit):
    """Test valid initialization with different units."""
    voxel_size = (0.5, 0.5, 0.5)
    vs = VoxelSize(voxels_size=voxel_size, unit=unit)
    assert vs.voxels_size == voxel_size
    assert vs.unit == "um"


@pytest.mark.parametrize(
    "voxel_size, unit",
    [
        ((-1.0, 1.0, 1.0), "um"),
        ((0.0, 1.0, 1.0), "um"),
        ((1.0, 1.0, 1.0), "nm"),  # invalid unit
        ((0.2, 0.1, 0.1, 0.1), "um"),  # too many values in voxel_size
        ((0.2, 0.1), "um"),  # too few values in voxel_size
    ],
)
def test_voxel_size_invalid_initialization(voxel_size, unit):
    """Test invalid initializations for voxel size and units."""
    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=voxel_size, unit=unit)


def test_voxel_size_equality():
    """Test equality and inequality of VoxelSize objects."""
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    same_voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    different_voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")

    assert same_voxel_size == voxel_size, (
        "VoxelSize objects should be equal if attributes match"
    )
    assert different_voxel_size != voxel_size, (
        "VoxelSize objects should be different if attributes differ"
    )


def test_voxel_size_scalefactor_from_voxelsize():
    """Test scale factor calculation between two VoxelSize objects."""
    vs1 = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    vs2 = VoxelSize(voxels_size=(2.0, 2.0, 2.0), unit="um")

    scale_factor = vs1.scalefactor_from_voxelsize(vs2)
    assert scale_factor == (0.5, 0.5, 0.5), (
        "Double voxel size should result in half the scale factor"
    )

    vs_missing = VoxelSize(unit="um")
    with pytest.raises(ValueError):
        vs1.scalefactor_from_voxelsize(vs_missing)


def test_voxel_size_voxelsize_from_factor():
    """Test voxel size adjustment based on a scaling factor."""
    vs = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    factor = (2.0, 2.0, 2.0)

    new_vs = vs.voxelsize_from_factor(factor)
    assert new_vs.voxels_size == (0.5, 0.5, 0.5), (
        "Doubling resolution should halve voxel size"
    )
    assert new_vs.unit == "um"

    vs_invalid = VoxelSize(unit="um")
    with pytest.raises(ValueError):
        vs_invalid.voxelsize_from_factor(factor)


def test_voxel_size_properties():
    """Test properties of VoxelSize object for x, y, z access."""
    vs = VoxelSize(voxels_size=(3.0, 2.0, 1.0), unit="um")

    assert vs.x == 1.0
    assert vs.y == 2.0
    assert vs.z == 3.0

    vs_empty = VoxelSize(unit="um")
    assert vs_empty.x == 1.0
    assert vs_empty.y == 1.0
    assert vs_empty.z == 1.0


def test_voxel_size_iter():
    """Test VoxelSize object's iterability."""
    vs = VoxelSize(voxels_size=(1.0, 2.0, 3.0), unit="um")
    assert list(vs) == [1.0, 2.0, 3.0]

    with pytest.raises(TypeError):
        vs[2]  # VoxelSize object is not subscriptable

    for v in vs:
        assert v in [1.0, 2.0, 3.0]


def test_voxel_size_array():
    """Test conversion of VoxelSize object to numpy array."""
    vs = VoxelSize(voxels_size=(1.0, 2.0, 3.0), unit="um")
    np.testing.assert_allclose(vs, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        np.array(VoxelSize(unit="um"))
