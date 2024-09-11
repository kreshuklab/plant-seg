from pathlib import Path

from plantseg.core.image import PlantSegImage, import_image, save_image
from plantseg.tasks import task_tracker
from plantseg.tasks.workflow_handler import TaskUserInput


@task_tracker(
    is_root=True,
    list_private_params=["semantic_type", "stack_layout"],
    list_inputs={
        "input_path": TaskUserInput(
            allowed_types=['str', 'list[str]'],
            description="Path to a file, or a directory containing files (all files will be imported) or list of paths.",
            headless_default=None,
            user_input_required=True,
        ),
        "image_name": TaskUserInput(
            allowed_types=['None', 'str'],
            description="Name of the image (if None, the file name will be used)",
            headless_default=None,
            user_input_required=False,
        ),
    },
)
def import_image_task(
    input_path: Path,
    semantic_type: str,
    stack_layout: str,
    image_name: str | None = None,
    key: str | None = None,
    m_slicing: str | None = None,
) -> PlantSegImage:
    """
    Task wrapper creating a PlantSegImage object from an image file.

    Args:
        input_path (Path): path to the image file
        semantic_type (str): semantic type of the image (raw, segmentation, prediction)
        stack_layout (str): stack layout of the image (3D, 2D, 2D_time)
        image_name (str | None): name of the image (if None, the file name will be used)
        key (str | None): key for the image (used only for h5 and zarr formats)
        m_slicing (str | None): m_slicing of the image (None, time, z, y, x)
    """

    if image_name is None:
        image_name = input_path.stem

    return import_image(
        path=input_path,
        key=key,
        image_name=image_name,
        semantic_type=semantic_type,
        stack_layout=stack_layout,
        m_slicing=m_slicing,
    )


@task_tracker(
    is_leaf=True,
    list_inputs={
        "output_directory": TaskUserInput(
            allowed_types=['str'],
            description="Output directory path where the image will be saved",
            headless_default=None,
            user_input_required=True,
        ),
        "output_file_name": TaskUserInput(
            allowed_types=['str', 'None'],
            description="Output file name (if None, the image name will be used)",
            headless_default=None,
            user_input_required=False,
        ),
    },
)
def export_image_task(
    image: PlantSegImage,
    output_directory: Path,
    output_file_name: str | None = None,
    custom_key_suffix: str | None = None,
    scale_to_origin: bool = True,
    file_format: str = "tiff",
    dtype: str = "uint16",
) -> None:
    """
    Task wrapper for saving an PlantSegImage object to disk.

    Args:
        image (PlantSegImage): input image to be saved to disk
        output_directory (Path): output directory path where the image will be saved
        output_file_name (str | None): output file name (if None, the image name will be used)
        custom_key_suffix (str | None): custom key for the image. If format is .h5 or .zarr this key will be used
            to create the dataset. If None, the semantic type will be used (raw, segmentation, predictio).
            If the image is tiff, the custom key will be added to the file name as a suffix.
            If None, the custom key will not be added.
        scale_to_origin (bool): scale to origin
        file_format (str): file format
        dtype (str): data type
    """
    if output_file_name is None:
        output_file_name = image.name

    if custom_key_suffix is None:
        if file_format == "tiff":
            custom_key_suffix = ""
        else:
            custom_key_suffix = image.semantic_type.value

    save_image(
        image=image,
        directory=output_directory,
        file_name=output_file_name,
        custom_key=custom_key_suffix,
        scale_to_origin=scale_to_origin,
        file_format=file_format,
        dtype=dtype,
    )
    return None
