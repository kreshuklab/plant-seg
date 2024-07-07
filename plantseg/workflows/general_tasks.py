from plantseg.workflows import task_tracker
from plantseg.dataprocessing import image_gaussian_smoothing
from plantseg.image import Image, import_image, save_image
from pathlib import Path


@task_tracker(
    is_root=True,
    list_private_params=["image_type", "stack_layout"],
    list_inputs=["input_path"],
)
def import_image_workflow(
    input_path: Path,
    key: str,
    image_name: str,
    image_type: str,
    stack_layout: str,
) -> Image:
    return import_image(
        path=input_path,
        key=key,
        image_name=image_name,
        image_type=image_type,
        stack_layout=stack_layout,
    )


@task_tracker(is_leaf=True, list_inputs=["output_directory", "output_file_name"])
def export_image_workflow(
    image: Image,
    output_directory: Path,
    output_file_name: str,
    custom_key: str,
    scale_to_origin: bool,
    file_format: str = "tiff",
    dtype: str = "uint16",
) -> None:
    save_image(
        image=image,
        directory=output_directory,
        file_name=output_file_name,
        custom_key=custom_key,
        scale_to_origin=scale_to_origin,
        file_format=file_format,
        dtype=dtype,
    )
    return None


@task_tracker
def gaussian_smoothing_workflow(image: Image, sigma: float) -> Image:
    data = image.data
    smoothed_data = image_gaussian_smoothing(data, sigma=sigma)
    new_image = image.derive_new(smoothed_data, name=f"{image.name}_smoothed")
    return new_image


@task_tracker(is_multioutput=True)
def mock_task1(image: Image) -> tuple[Image, Image]:
    image2 = image.derive_new(image.data, name=f"{image.name}_m1")
    image3 = image.derive_new(image.data, name=f"{image.name}_m2")
    return image2, image3


@task_tracker
def mock_task2(image: Image, image2: Image) -> Image:
    return image.derive_new(image.data, name=f"{image.name}_m3")
