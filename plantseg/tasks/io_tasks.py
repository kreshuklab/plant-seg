from pathlib import Path

from plantseg.core.image import PlantSegImage, import_image, save_image
from plantseg.tasks import task_tracker
from plantseg.tasks.workflow_handler import RunTimeInputSchema


@task_tracker(
    is_root=True,
    list_inputs={
        "input_path": RunTimeInputSchema(
            description="Path to a file, or a directory containing files (all files will be imported) or list of paths.",
            required=True,
            is_input_file=True,
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
        image_name (str): name of the image, if None the name will be the same as the file name
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
        "export_directory": RunTimeInputSchema(
            description="Output directory path where the image will be saved",
            required=True,
        ),
        "name_pattern": RunTimeInputSchema(
            description="Output file name pattern. Can contain the special {image_name} or {file_name} tokens ",
            required=False,
        ),
    },
)
def export_image_task(
    image: PlantSegImage,
    export_directory: Path,
    name_pattern: str = "{file_name}_export",
    key: str | None = None,
    scale_to_origin: bool = True,
    export_format: str = "tiff",
    data_type: str = "uint16",
) -> None:
    """
    Task wrapper for saving an PlantSegImage object to disk.

    Args:
        image (PlantSegImage): input image to be saved to disk
        export_directory (Path): output directory path where the image will be saved
        name_pattern (str): output file name pattern, can contain the {image_name} or {file_name} tokens
            to be replaced in the final file name.
        key (str | None): key for the image (used only for h5 and zarr formats).
        scale_to_origin (bool): scale the voxel size to the original one
        export_format (str): file format (tiff, h5, zarr)
        data_type (str): data type to save the image.
    """
    save_image(
        image=image,
        export_directory=export_directory,
        name_pattern=name_pattern,
        key=key,
        scale_to_origin=scale_to_origin,
        export_format=export_format,
        data_type=data_type,
    )
    return None
