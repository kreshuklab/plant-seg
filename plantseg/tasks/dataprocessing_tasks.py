from plantseg.tasks import task_tracker
from plantseg.dataprocessing import image_gaussian_smoothing
from plantseg.image import PlantSegImage
from plantseg.io.utils import VoxelSize
import numpy as np
from plantseg.dataprocessing import (
    image_rescale,
    compute_scaling_voxelsize,
    scale_image_to_voxelsize,
)


@task_tracker
def gaussian_smoothing_task(image: PlantSegImage, sigma: float) -> PlantSegImage:
    """
    Apply Gaussian smoothing to a PlantSegImage object.

    Args:
        image (PlantSegImage): input image
        sigma (float): standard deviation of the Gaussian kernel

    """
    data = image.data
    smoothed_data = image_gaussian_smoothing(data, sigma=sigma)
    new_image = image.derive_new(smoothed_data, name=f"{image.name}_smoothed")
    return new_image


@task_tracker
def set_voxel_size_task(image: PlantSegImage, voxel_size: tuple[float, float, float]) -> PlantSegImage:
    new_voxel_size = VoxelSize(voxels_size=voxel_size)
    new_image = image.derive_new(
        image.data, name=image.name, voxel_size=new_voxel_size, original_voxel_size=new_voxel_size
    )
    return new_image


@task_tracker
def image_rescale_to_shape_task(image: PlantSegImage, new_shape: tuple[int, int, int], order: int = 0) -> PlantSegImage:
    scaling_factor = np.array(new_shape) / np.array(image.data.shape)
    out_data = image_rescale(image.data, scaling_factor, order=order)

    if image.has_valid_voxel_size():
        _out_voxel_size = compute_scaling_voxelsize(image.voxel_size.voxels_size, scaling_factor)
        out_voxel_size = VoxelSize(voxels_size=_out_voxel_size)
    else:
        out_voxel_size = VoxelSize()

    new_image = image.derive_new(out_data, name=f"{image.name}_reshaped", voxel_size=out_voxel_size)
    return new_image


@task_tracker
def image_rescale_to_voxel_size_task(
    image: PlantSegImage, new_voxel_size: tuple[float, float, float], order: int = 0
) -> PlantSegImage:
    out_data = scale_image_to_voxelsize(image.data, image.voxel_size.voxels_size, new_voxel_size, order=order)
    out_voxel_size = VoxelSize(voxels_size=new_voxel_size)
    new_image = image.derive_new(out_data, name=f"{image.name}_rescaled", voxel_size=out_voxel_size)
    return new_image
