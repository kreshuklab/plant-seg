from pydantic import BaseModel
import numpy as np
from typing import Protocol, Optional
from pathlib import Path


class VoxelSize(BaseModel):
    """
    Voxel size of an image.

    Attributes:
        voxels_size (tuple[float]): Size of the voxels in the image
        unit (str): Unit of the voxel size
    """

    voxels_size: Optional[tuple[float, ...]] = None
    unit: str = "um"
