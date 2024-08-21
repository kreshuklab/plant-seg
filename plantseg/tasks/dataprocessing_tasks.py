from plantseg.functionals.dataprocessing import (
    image_gaussian_smoothing,
    image_rescale,
)
from plantseg.io.utils import VoxelSize
from plantseg.plantseg_image import ImageLayout, PlantSegImage
from plantseg.tasks import task_tracker


@task_tracker
def gaussian_smoothing_task(image: PlantSegImage, sigma: float) -> PlantSegImage:
    """
    Apply Gaussian smoothing to a PlantSegImage object.

    Args:
        image (PlantSegImage): input image
        sigma (float): standard deviation of the Gaussian kernel

    """
    if image.is_multichannel:
        raise ValueError("Gaussian smoothing is not supported for multichannel images.")

    data = image.get_data()
    smoothed_data = image_gaussian_smoothing(data, sigma=sigma)
    new_image = image.derive_new(smoothed_data, name=f"{image.name}_smoothed")
    return new_image


@task_tracker
def set_voxel_size_task(image: PlantSegImage, voxel_size: tuple[float, float, float]) -> PlantSegImage:
    """Set the voxel size of an image.

    Args:
        image (PlantSegImage): input image
        voxel_size (tuple[float, float, float]): new voxel size

    """
    new_voxel_size = VoxelSize(voxels_size=voxel_size)
    new_name = f"{image.name}_set_voxel_size"
    new_image = image.derive_new(
        image._data, name=new_name, voxel_size=new_voxel_size, original_voxel_size=new_voxel_size
    )
    return new_image


@task_tracker
def image_rescale_to_shape_task(image: PlantSegImage, new_shape: tuple[int, ...], order: int = 0) -> PlantSegImage:
    """Rescale an image to a new shape.

    Args:
        image (PlantSegImage): input image
        new_shape (tuple[int, ...]): new shape of the image
        order (int): order of the interpolation
    """
    if image.image_layout == ImageLayout.YX:
        scaling_factor = (new_shape[1] / image.shape[0], new_shape[2] / image.shape[1])
        spatial_scaling_factor = (1.0, scaling_factor[0], scaling_factor[1])
    elif image.image_layout == ImageLayout.ZYX:
        scaling_factor = (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1], new_shape[2] / image.shape[2])
        spatial_scaling_factor = scaling_factor
    elif image.image_layout == ImageLayout.CYX:
        scaling_factor = (1.0, new_shape[1] / image.shape[1], new_shape[2] / image.shape[2])
        spatial_scaling_factor = (1.0, scaling_factor[1], scaling_factor[2])
    elif image.image_layout == ImageLayout.CZYX:
        scaling_factor = (
            1.0,
            new_shape[0] / image.shape[1],
            new_shape[1] / image.shape[2],
            new_shape[2] / image.shape[3],
        )
        spatial_scaling_factor = scaling_factor[1:]
    elif image.image_layout == ImageLayout.ZCYX:
        scaling_factor = (
            new_shape[0] / image.shape[0],
            1.0,
            new_shape[1] / image.shape[2],
            new_shape[2] / image.shape[3],
        )
        spatial_scaling_factor = (scaling_factor[0], scaling_factor[2], scaling_factor[3])

    out_data = image_rescale(image.get_data(), scaling_factor, order=order)

    if image.has_valid_voxel_size():
        out_voxel_size = image.voxel_size.voxelsize_from_factor(spatial_scaling_factor)
    else:
        out_voxel_size = VoxelSize()

    new_image = image.derive_new(out_data, name=f"{image.name}_reshaped", voxel_size=out_voxel_size)
    return new_image


@task_tracker
def image_rescale_to_voxel_size_task(image: PlantSegImage, new_voxel_size: VoxelSize, order: int = 0) -> PlantSegImage:
    """Rescale an image to a new voxel size.
    If the voxel size is not defined in the input image, use the set voxel size task to set the voxel size.
    Args:
        image (PlantSegImage): input image
        new_voxel_size (VoxelSize): new voxel size
        order (int): order of the interpolation

    """
    spatial_scaling_factor = image.voxel_size.scalefactor_from_voxelsize(new_voxel_size)

    if image.image_layout == ImageLayout.YX:
        scaling_factor = (spatial_scaling_factor[1], spatial_scaling_factor[2])
    elif image.image_layout == ImageLayout.CYX:
        scaling_factor = (1.0, spatial_scaling_factor[1], spatial_scaling_factor[2])
    elif image.image_layout == ImageLayout.ZYX:
        scaling_factor = spatial_scaling_factor
    elif image.image_layout == ImageLayout.CZYX:
        scaling_factor = (1.0, *spatial_scaling_factor)
    elif image.image_layout == ImageLayout.ZCYX:
        scaling_factor = (spatial_scaling_factor[0], 1.0, *spatial_scaling_factor[1:])

    out_data = image_rescale(image.get_data(), scaling_factor, order=order)
    new_image = image.derive_new(out_data, name=f"{image.name}_rescaled", voxel_size=new_voxel_size)
    return new_image
