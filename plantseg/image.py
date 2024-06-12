from typing import Protocol, Optional, Any, Callable
import numpy as np
from pydantic import BaseModel
from enum import Enum
from plantseg.io.utils import VoxelSize
from pathlib import Path
from plantseg.io import load_h5, load_pil, load_tiff, load_zarr
from plantseg.io import read_h5_voxel_size, read_tiff_voxel_size, read_zarr_voxel_size
from plantseg.io import PIL_EXTENSIONS, H5_EXTENSIONS, ZARR_EXTENSIONS, TIFF_EXTENSIONS, allowed_data_format


class ImageType(Enum):
    """
    Enum class for image types.

    Attributes:
        RAW (str): Reserved for raw images (e.g. microscopy images)
        SEGMENTATION (str): Reserved for segmentation masks
        PREDICTION (str): Reserved for model predictions
        LABEL (str): Reserved for ground truth labels
    """

    RAW = "raw"
    SEGMENTATION = "segmentation"
    PREDICTION = "prediction"
    LABEL = "label"


class ImageDimensionality(Enum):
    """
    Enum class for image dimensionality.

    Attributes:
        TWO (str): 2D images
        THREE (str): 3D images
    """

    TWO = "2D"
    THREE = "3D"


class ImageLayout(Enum):
    XY = ("XY", ImageDimensionality.TWO, None)
    CXY = ("CXY", ImageDimensionality.TWO, 0)
    ZXY = ("ZXY", ImageDimensionality.THREE, None)
    CZXY = ("CZXY", ImageDimensionality.THREE, 0)
    ZCXY = ("ZCXY", ImageDimensionality.THREE, 1)
    UNKNOWN = ("UNKNOWN", ImageDimensionality.THREE, None)

    def __init__(self, layout: str, dimensionality: ImageDimensionality, channel_axis: Optional[int] = None):
        self.layout = layout
        self.dimensionality = dimensionality
        self.channel_axis = channel_axis

    @classmethod
    def to_choices(cls) -> list[str]:
        return [il.layout for il in cls]


class ImageProperties(BaseModel):
    """
    Basic properties of an image.

    Attributes:
        name (str): Current name to identify the image
        image_type (ImageType): Type of image
        voxel_size (VoxelSize): Voxel size of the image
        image_layout (ImageLayout): Layout of the image
        root_name (Optional[str]): Name of the root image from which the current image was derived
        root_voxel_size (Optional[VoxelSize]): Voxel size of the root image from which the current image was derived
    """

    name: str
    image_type: ImageType
    voxel_size: VoxelSize
    image_layout: ImageLayout

    root_name: Optional[str] = None
    root_voxel_size: Optional[VoxelSize] = None


class Image(BaseModel):
    data: np.ndarray
    properties: ImageProperties


def get_data(path: Path, key: str) -> tuple[np.ndarray, VoxelSize]:
    ext = path.suffix

    if ext not in allowed_data_format:
        raise ValueError(f"File extension is {ext} but should be one of {allowed_data_format}")

    if ext in H5_EXTENSIONS:
        h5_key = key if key else None
        return load_h5(path, key), read_h5_voxel_size(path, h5_key)

    elif ext in TIFF_EXTENSIONS:
        return load_tiff(path), read_tiff_voxel_size(path)

    elif ext in PIL_EXTENSIONS:
        return load_pil(path), VoxelSize()

    elif ext in ZARR_EXTENSIONS:
        zarr_key = key if key else None
        return load_zarr(path, key), read_zarr_voxel_size(path, zarr_key)

    else:
        raise NotImplementedError()


def create_image(
    path: Path,
    key: str,
    image_name: str,
    stack_layout: ImageLayout,
    m_slicing: Optional[str] = None,
    layer_type="image",
) -> Image:

    data, voxel_size = get_data(path, key)
    image_properties = ImageProperties(
        name=image_name, image_type=ImageType.RAW, voxel_size=voxel_size, image_layout=stack_layout
    )
    # TODO: select the correct slicing, channel and checks
    return Image(data=data, properties=image_properties)


def save_image(image: Image, path: Path, key: str) -> None:
    # TODO scaling and normalization
    pass


def apply_function_to_image(
    func: Callable,
    run_time_inputs: dict[str, Image],
    static_kwargs: dict[str, Any],
    out_properties: list[ImageProperties],
) -> Image | tuple[Image]:
    _run_time_inputs = {key: image.data for key, image in run_time_inputs.items()}
    data = func(**_run_time_inputs, **static_kwargs)

    if isinstance(data, np.ndarray):
        data = (data,)

    assert len(data) == len(out_properties), "Number of outputs does not match the number of properties"

    returns = []
    for _data, _properties in zip(data, out_properties):
        returns.append(Image(data=_data, properties=_properties))

    return tuple(returns)
