"""Voxel size of an image.

This module handles voxel size operations to avoid circular imports with plantseg.core.image.
"""

import logging

import numpy as np
from pydantic import BaseModel, Field, field_validator

from plantseg.functionals.dataprocessing import (
    compute_scaling_factor,
    compute_scaling_voxelsize,
)

logger = logging.getLogger(__name__)


class VoxelSize(BaseModel):
    """
    Voxel size of an image.

    Attributes:
        voxels_size (tuple[float, float, float] | None): Size of the voxels in the image.
        unit (str): Unit of the voxel size, restricted to micrometers (um).
    """

    voxels_size: tuple[float, float, float] | None = None
    unit: str = Field(default="um")

    @field_validator("voxels_size")
    @classmethod
    def _check_voxel_size(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and any(v <= 0 for v in value):
            raise ValueError("Voxel size must be positive")
        return value

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, value: str) -> str:
        if value.lower().startswith(
            (
                "u",
                "\u03bc",  # Unicode characters for "mu", i.e. 'μ'.
                "\\u03bc",  # `tifffile` uses raw string.
                "\u00b5",  # Unicode characters for "micro", i.e. 'µ', to which "mu" is converted in Fiji by default.
                "\\u00b5",  # `tifffile` uses raw string.
                "micro",
            )
        ):
            return "um"
        elif value.lower() in ["-", ""]:
            logger.warning("Unit is not defined, assuming micrometers (um)")
            return "um"
        raise ValueError("Only micrometers (um) are supported")

    @property
    def x(self) -> float:
        """Voxel size in the x direction, or 1.0 if not defined."""
        return self.voxels_size[2] if self.voxels_size else 1.0  # pylint: disable=unsubscriptable-object

    @property
    def y(self) -> float:
        """Voxel size in the y direction, or 1.0 if not defined."""
        return self.voxels_size[1] if self.voxels_size else 1.0  # pylint: disable=unsubscriptable-object

    @property
    def z(self) -> float:
        """Voxel size in the z direction, or 1.0 if not defined."""
        return self.voxels_size[0] if self.voxels_size else 1.0  # pylint: disable=unsubscriptable-object

    def __len__(self) -> int:
        """Return the number of dimensions of the voxel size."""
        if self.voxels_size is None:
            raise ValueError("Voxel size must be defined to get the length")
        return len(self.voxels_size)

    def __iter__(self):
        """Allow the VoxelSize instance to be unpacked as a tuple."""
        if self.voxels_size is None:
            raise ValueError("Voxel size must be defined to iterate")
        return iter(self.voxels_size)

    def __array__(self):
        if self.voxels_size is None:
            raise ValueError("Voxel size is not defined")
        return np.array(self.voxels_size)

    def as_tuple(self) -> tuple[float, float, float]:
        """Convert VoxelSize to a tuple."""
        if self.voxels_size is None:
            raise ValueError("Voxel size must be defined to convert to tuple")
        return self.voxels_size

    def scalefactor_from_voxelsize(
        self, other: "VoxelSize"
    ) -> tuple[float, float, float]:
        """Compute the scaling factor to rescale an image from the current voxel size to another."""
        if self.voxels_size is None or other.voxels_size is None:
            raise ValueError(
                "Both voxel sizes must be defined to compute the scaling factor"
            )
        return compute_scaling_factor(self.voxels_size, other.voxels_size)

    def voxelsize_from_factor(self, factor: tuple[float, float, float]) -> "VoxelSize":
        """Compute the voxel size after scaling with the given factor."""
        if self.voxels_size is None:
            raise ValueError(
                "Voxel size must be defined to compute the output voxel size"
            )
        return VoxelSize(
            voxels_size=compute_scaling_voxelsize(self.voxels_size, factor)
        )
