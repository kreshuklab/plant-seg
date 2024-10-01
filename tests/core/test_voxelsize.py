import numpy as np
import pytest
from pydantic import ValidationError

from plantseg.io.voxelsize import VoxelSize


def test_voxel_size_initialization():
    # Test valid initialization
    vs = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    assert vs.voxels_size == (1.0, 1.0, 1.0)
    assert vs.unit == "um"

    # Test valid initialization with different unit format
    vs = VoxelSize(voxels_size=(0.5, 0.5, 0.5), unit="µm")
    assert vs.voxels_size == (0.5, 0.5, 0.5)
    assert vs.unit == "um"

    vs = VoxelSize(voxels_size=(0.5, 0.5, 0.5), unit="micrometer")
    assert vs.unit == "um"


def test_voxel_size_invalid_initialization():
    # Test negative voxel size
    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=(-1.0, 1.0, 1.0), unit="um")

    # Test zero voxel size
    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=(0.0, 1.0, 1.0), unit="um")

    # Test invalid unit
    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="nm")

    # Voxel size with more than >3 or <3 values should raise an error
    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=(0.2, 0.1, 0.1, 0.1))

    with pytest.raises(ValidationError):
        VoxelSize(voxels_size=(0.2, 0.1))


def test_voxel_size_equality():
    voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    same_voxel_size = VoxelSize(voxels_size=(0.5, 1.0, 1.0), unit="um")
    original_voxel_size = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    assert same_voxel_size == voxel_size, "Pydanctic models should be equal if their attributes are equal"
    assert original_voxel_size != voxel_size, "Pydanctic models should not be equal if their attributes are not equal"


def test_voxel_size_scalefactor_from_voxelsize():
    vs1 = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    vs2 = VoxelSize(voxels_size=(2.0, 2.0, 2.0), unit="um")

    scale_factor = vs1.scalefactor_from_voxelsize(vs2)
    assert scale_factor == (0.5, 0.5, 0.5), "Note that double the voxel size means half the resolution"

    # Test missing voxel size in one of the VoxelSize objects
    vs3 = VoxelSize(unit="um")
    with pytest.raises(ValueError):
        vs1.scalefactor_from_voxelsize(vs3)


def test_voxel_size_voxelsize_from_factor():
    """Note that double the resolution means half the voxel size"""
    vs = VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit="um")
    factor = (2.0, 2.0, 2.0)

    new_vs = vs.voxelsize_from_factor(factor)
    assert new_vs.voxels_size == (0.5, 0.5, 0.5), "Note that double the resolution means half the voxel size"
    assert new_vs.unit == "um"

    # Test missing voxel size
    vs_invalid = VoxelSize(unit="um")
    with pytest.raises(ValueError):
        vs_invalid.voxelsize_from_factor(factor)


def test_voxel_size_properties():
    vs = VoxelSize(voxels_size=(3.0, 2.0, 1.0), unit="um")

    assert vs.x == 1.0
    assert vs.y == 2.0
    assert vs.z == 3.0

    vs_empty = VoxelSize(unit="um")
    assert vs_empty.x == 1.0
    assert vs_empty.y == 1.0
    assert vs_empty.z == 1.0


def test_voxel_size_iter():
    vs = VoxelSize(voxels_size=(1.0, 2.0, 3.0), unit="um")
    assert list(vs) == [1.0, 2.0, 3.0]

    with pytest.raises(TypeError):
        assert vs[2] == 3.0, "VoxelSize object is not subscriptable, encouraging users to use .x, .y, .z properties"

    for v in vs:
        assert v in vs, "VoxelSize object should be iterable"

    vs_empty = VoxelSize(unit="um")
    with pytest.raises(ValueError):
        list(vs_empty)


def test_voxel_size_array():
    """Test the __array__ method"""
    vs = VoxelSize(voxels_size=(1.0, 2.0, 3.0), unit="um")
    np.testing.assert_allclose(vs, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        np.array(VoxelSize(unit="um"))
