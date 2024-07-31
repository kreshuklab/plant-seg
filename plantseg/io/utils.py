"""Utility functions and classes for I/O operations."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def compute_scaling_factor(
    input_voxel_size: tuple[float, float, float], output_voxel_size: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Compute the scaling factor to rescale an image from input voxel size to output voxel size.
    """
    scaling = tuple(i_size / o_size for i_size, o_size in zip(input_voxel_size, output_voxel_size))
    if len(scaling) != 3:
        raise ValueError(f"Expected scaling factor to be 3D, but got {len(scaling)}D input")
    return scaling


def compute_scaling_voxelsize(
    input_voxel_size: tuple[float, float, float], scaling_factor: tuple[float, float, float]
) -> tuple[float, float, float]:
    """
    Compute the output voxel size after scaling an image with a given scaling factor.
    """
    output_voxel_size = tuple(i_size / s_size for i_size, s_size in zip(input_voxel_size, scaling_factor))
    if len(output_voxel_size) != 3:
        raise ValueError(f"Expected output voxel size to be 3D, but got {len(output_voxel_size)}D input")
    return output_voxel_size


class VoxelSize(BaseModel):
    """
    Voxel size of an image.

    Attributes:
        voxels_size (Optional[tuple[float, float, float]]): Size of the voxels in the image.
        unit (Literal["um", "µm", "micron", "micrometers"]): Unit of the voxel size.
    """

    voxels_size: Optional[tuple[float, float, float]] = None
    unit: Literal["um", "µm", "micron", "micrometers"] = Field(default="um")

    @field_validator("voxels_size")
    @classmethod
    def _check_voxel_size(cls, values: Optional[tuple[float, float, float]]):
        if values is None:
            return values

        if any(value <= 0 for value in values):
            raise ValueError("Voxel size must be positive")

        return values

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, value: str) -> str:
        if value in ["um", "µm", "micron", "micrometers"]:
            return "um"
        raise ValueError("Only micrometers (um) are supported as unit")

    def scalefactor_from_voxelsize(self, other: "VoxelSize") -> tuple[float, float, float]:
        """
        Compute the scaling factor to rescale an image from the current voxel size to another voxel size.
        """
        if self.voxels_size is None or other.voxels_size is None:
            raise ValueError("Voxel size is not defined, cannot compute scaling factor")

        return compute_scaling_factor(self.voxels_size, other.voxels_size)

    def voxelsize_from_factor(self, factor: tuple[float, float, float]) -> "VoxelSize":
        """
        Compute the output voxel size after scaling an image with a given scaling factor.
        """
        if self.voxels_size is None:
            raise ValueError("Voxel size is not defined, cannot compute output voxel size")

        return VoxelSize(voxels_size=compute_scaling_voxelsize(self.voxels_size, factor))

    @property
    def x(self) -> float:
        """Safe access to the voxel size in the x direction. Returns 1.0 if the voxel size is not defined."""
        if self.voxels_size is not None:
            return self.voxels_size[1]
        return 1.0

    @property
    def y(self) -> float:
        """Safe access to the voxel size in the y direction. Returns 1.0 if the voxel size is not defined."""
        if self.voxels_size is not None:
            return self.voxels_size[2]
        return 1.0

    @property
    def z(self) -> float:
        """Safe access to the voxel size in the z direction. Returns 1.0 if the voxel size is not defined."""
        if self.voxels_size is not None:
            return self.voxels_size[0]
        return 1.0

    @property
    def is_valid(self) -> bool:
        """Return True if the voxel size is valid (i.e., not None)."""
        return self.voxels_size is not None
