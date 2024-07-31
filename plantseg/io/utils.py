from pydantic import BaseModel, field_validator
from typing import Optional


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


class VoxelSize(BaseModel):
    """
    Voxel size of an image.

    Attributes:
        voxels_size (tuple[float]): Size of the voxels in the image
        unit (str): Unit of the voxel size
    """

    voxels_size: Optional[tuple[float, ...]] = None
    unit: str = "um"

    @field_validator("voxels_size")
    def _check_voxel_size(cls, values):
        if values is None:
            return None

        assert len(values) == 3, f"Expected voxel size to be 3d, but got {len(values)}d input"

        if any(value <= 0 for value in values):
            raise ValueError("Voxel size must be positive")

        return values

    @field_validator("unit")
    def _check_unit(cls, value):
        if value in ["um", "\\u00B5m", "micron", "micrometers"]:
            return "um"

        raise ValueError("Only micrometers (um) are supported as unit")

    def scalefactor_from_voxelsize(self, other: "VoxelSize") -> tuple[float, float, float]:
        """
        Compute the predicted scaling factor to rescale an image from the current voxel size to another voxel size.
        """
        if self.voxels_size is None or other.voxels_size is None:
            raise ValueError("Voxel size is not defined, cannot compute scaling factor")

        return compute_scaling_factor(self.voxels_size, other.voxels_size)

    def voxelsize_from_factor(self, factor: tuple[float, float, float]) -> "VoxelSize":
        """
        Compute the predicted output voxel size after scaling an image with a given scaling factor.
        """
        if self.voxels_size is None:
            raise ValueError("Voxel size is not defined, cannot compute output voxel size")

        return VoxelSize(voxels_size=compute_scaling_voxelsize(self.voxels_size, factor))

    @property
    def z(self) -> float:
        """Safe access to the voxel size in the z direction. return 1.0 if the voxel size is not defined."""
        if self.voxels_size is None:
            return 1.0
        return self.voxels_size[0]

    @property
    def x(self) -> float:
        """Safe access to the voxel size in the x direction. return 1.0 if the voxel size is not defined."""
        if self.voxels_size is None:
            return 1.0
        return self.voxels_size[1]

    @property
    def y(self) -> float:
        """Safe access to the voxel size in the y direction. return 1.0 if the voxel size is not defined."""
        if self.voxels_size is None:
            return 1.0
        return self.voxels_size[2]

    @property
    def is_valid(self) -> bool:
        """Return True if the voxel size is valid. i.e. not None."""
        return self.voxels_size is not None
