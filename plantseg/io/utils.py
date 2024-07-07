from pydantic import BaseModel
from typing import Optional


class VoxelSize(BaseModel):
    """
    Voxel size of an image.

    Attributes:
        voxels_size (tuple[float]): Size of the voxels in the image
        unit (str): Unit of the voxel size
    """

    voxels_size: Optional[tuple[float, ...]] = None
    unit: str = "um"
