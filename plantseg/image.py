from typing import Optional
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum
from plantseg.io.utils import VoxelSize
from pathlib import Path
import plantseg.dataprocessing as dp
from plantseg.io import load_h5, load_pil, load_tiff, load_zarr
from plantseg.io import create_h5, create_tiff, create_zarr
from napari.types import LayerDataTuple
from napari.layers import Image, Labels
from uuid import uuid4, UUID

from plantseg.io import (
    read_h5_voxel_size,
    read_tiff_voxel_size,
    read_zarr_voxel_size,
)
from plantseg.io import (
    PIL_EXTENSIONS,
    H5_EXTENSIONS,
    ZARR_EXTENSIONS,
    TIFF_EXTENSIONS,
    allowed_data_format,
)


class SemanticType(Enum):
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


class ImageType(Enum):
    """
    Enum class for image types.

    Attributes:
        IMAGE (str): Image data
        LABEL (str): Label data
    """

    IMAGE = "image"
    LABEL = "label"

    @classmethod
    def to_choices(cls):
        return [member.value for member in cls]


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
    XY = "XY"
    CXY = "CXY"
    ZXY = "ZXY"
    CZXY = "CZXY"
    ZCXY = "ZCXY"

    @classmethod
    def to_choices(cls) -> list[str]:
        return [il.layout for il in cls]


class ImageProperties(BaseModel):
    """
    Basic properties of an image.

    Attributes:
        name (str): Name of the image
        semantic_type (SemanticType): Semantic type of the image
        voxel_size (VoxelSize): Voxel size of the image
        image_layout (ImageLayout): Image layout of the image
        original_voxel_size (VoxelSize): Original voxel size of the image
    """

    name: str = Field(frozen=True)
    semantic_type: SemanticType = Field(frozen=True)
    voxel_size: VoxelSize = Field(frozen=True)
    image_layout: ImageLayout = Field(frozen=True)
    original_voxel_size: VoxelSize = Field(frozen=True)

    @property
    def dimensionality(self) -> ImageDimensionality:
        if self.image_layout in (ImageLayout.XY, ImageLayout.CXY):
            return ImageDimensionality.TWO
        elif self.image_layout in (
            ImageLayout.ZXY,
            ImageLayout.CZXY,
            ImageLayout.ZCXY,
        ):
            return ImageDimensionality.THREE
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

    @property
    def image_type(self) -> ImageType:
        if self.semantic_type in (SemanticType.RAW, SemanticType.PREDICTION):
            return ImageType.IMAGE
        elif self.semantic_type in (
            SemanticType.SEGMENTATION,
            SemanticType.LABEL,
        ):
            return ImageType.LABEL
        else:
            raise ValueError(f"Semantic type {self.semantic_type} not recognized")

    @property
    def channel_axis(self) -> int:
        if self.image_layout in (ImageLayout.CXY, ImageLayout.CZXY):
            return 0
        elif self.image_layout in (ImageLayout.ZCXY):
            return 1

        elif self.image_layout in (ImageLayout.XY, ImageLayout.ZXY):
            return None
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

    def interpolation_order(self, image_default=1) -> int:
        if self.image_type == ImageType.LABEL:
            return 0
        elif self.image_type == ImageType.IMAGE:
            return image_default
        else:
            raise ValueError(f"Image type {self.image_type} not recognized")


class PlantSegImage:
    """
    Image class represent an image with its metadata and data.
    """

    _data: np.ndarray
    _properties: ImageProperties

    def __init__(self, data: np.ndarray, properties: ImageProperties):
        self._data = data
        self._properties = properties
        self._check_shape(data)
        self._id = uuid4()

    def derive_new(self, data: np.ndarray, name: str, **kwargs) -> "PlantSegImage":
        """
        Derive a new image from the current image.

        The new image will have the same properties as the original image, except for the name and the properties passed as kwargs.

        args:
            data (np.ndarray): New data
            name (str): New name
            **kwargs: other properties to change

        Returns:
            PlantSegImage: New image
        """

        property_dict = self._properties.model_dump()

        if name == self.name:
            raise ValueError("New derived name should be different from the original")

        property_dict["name"] = name

        for key, value in kwargs.items():
            if key in property_dict:
                property_dict[key] = value
            else:
                raise ValueError(f"Property {key} not recognized, should be one of {property_dict.keys()}")

        new_properties = ImageProperties(**property_dict)
        return PlantSegImage(data, new_properties)

    @classmethod
    def from_napari_layer(cls, layer: Image | Labels) -> "PlantSegImage":
        """
        Load a PlantSegImage from a napari layer.

        Args:
            layer (Image | Labels): Napari layer to load
        """

        metadata = layer.metadata

        if "semantic_type" not in metadata:
            raise ValueError("Semantic type not found in metadata")

        semantic_type = SemanticType(metadata["semantic_type"])

        if isinstance(layer, Image):
            image_type = ImageType.IMAGE
        elif isinstance(layer, Labels):
            image_type = ImageType.LABEL
        else:
            raise ValueError("Layer should be either Image or Labels")

        if "original_voxel_size" not in metadata:
            raise ValueError("Original voxel size not found in metadata")

        original_voxel_size = VoxelSize(**metadata["original_voxel_size"])

        if "image_layout" not in metadata:
            raise ValueError("Image layout not found in metadata")

        image_layout = ImageLayout(metadata["image_layout"])

        new_voxel_size = VoxelSize(voxels_size=layer.scale)

        # Loading from napari layer, the id needs to be present in the metadata
        # If not present, the layer is corrupted
        if "id" in metadata:
            id = metadata["id"]
        else:
            raise ValueError("ID not found in metadata")

        properties = ImageProperties(
            name=layer.name,
            semantic_type=semantic_type,
            voxel_size=new_voxel_size,
            image_layout=image_layout,
            original_voxel_size=original_voxel_size,
        )

        if image_type != properties.image_type:
            raise ValueError(f"Image type {image_type} does not match semantic type {properties.semantic_type}")

        ps_image = cls(layer.data, properties)
        ps_image._id = id
        return ps_image

    def to_napari_layer_tuple(self) -> LayerDataTuple:
        """
        Prepare the image to be loaded as a napari layer.
        All the metadata will be stored in the metadata of the layer.

        Returns:
            LayerDataTuple: Tuple containing the data, metadata and type of the image
        """
        name = self.name
        scale = self.voxel_size.voxels_size
        metadata = self._properties.model_dump()

        # When going to we need to preserve the id
        metadata["id"] = self.id
        return (
            self.data,
            {"name": name, "scale": scale, "metadata": metadata},
            self.image_type.value,
        )

    def _check_shape(self, data: np.ndarray) -> None:
        if self.image_layout in (ImageLayout.CXY, ImageLayout.ZXY):
            if data.ndim != 3:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 3 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.CZXY, ImageLayout.ZCXY):
            if data.ndim != 4:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 4 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.XY,):
            if data.ndim == 2:
                return None

            elif data.ndim == 3 and data.shape[0] == 1:
                return None

            else:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 2 dimensions for layout {self.image_layout}"
                )

        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

    @property
    def requires_scaling(self) -> bool:
        if self.voxel_size != self.original_voxel_size:
            return True
        return False

    @property
    def data(self) -> np.ndarray:
        if self.image_type == ImageType.LABEL:
            return self._data

        _data = dp.normalize_01(self._data)
        return _data

    @property
    def voxel_size(self) -> VoxelSize:
        return self._properties.voxel_size

    @property
    def original_voxel_size(self) -> VoxelSize:
        return self._properties.original_voxel_size

    @property
    def name(self) -> str:
        return self._properties.name

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def unique_name(self) -> str:
        return f"{self.name}_{self.id}"

    @property
    def image_type(self) -> ImageType:
        return self._properties.image_type

    @property
    def semantic_type(self) -> SemanticType:
        return self._properties.semantic_type

    @property
    def image_layout(self) -> ImageLayout:
        return self._properties.image_layout

    @property
    def dimensionality(self) -> ImageDimensionality:
        return self._properties.dimensionality

    @property
    def channel_axis(self) -> int:
        return self._properties.channel_axis

    def has_valid_voxel_size(self) -> bool:
        """
        Returns True if the voxel size is valid (not None), False otherwise.
        """
        if self.voxel_size.voxels_size is None:
            return False
        return True

    def has_valid_original_voxel_size(self) -> bool:
        """
        Returns True if the original voxel size is valid (not None), False otherwise.
        """
        if self.original_voxel_size.voxels_size is None:
            return False
        return True


def _load_data(path: Path, key: str) -> tuple[np.ndarray, VoxelSize]:
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


def _select_channel(data: np.ndarray, channel: int, image_layout: ImageLayout) -> tuple[np.ndarray | ImageLayout]:
    if image_layout == ImageLayout.CXY:
        return dp.select_channel(data, channel, channel_axis=0), ImageLayout.XY

    if image_layout == ImageLayout.CZXY:
        return (
            dp.select_channel(data, channel, channel_axis=0),
            ImageLayout.ZXY,
        )

    if image_layout == ImageLayout.ZCXY:
        return (
            dp.select_channel(data, channel, channel_axis=1),
            ImageLayout.ZXY,
        )

    return data, image_layout


def _add_singletons(data: np.ndarray, image_layout: ImageLayout) -> np.ndarray:
    if image_layout == ImageLayout.XY:
        return data[np.newaxis, ...]
    return data


def import_image(
    path: Path,
    key: str | None = None,
    image_name: str = "image",
    semantic_type: str = "raw",
    stack_layout: str = "XY",
    channel: int | None = None,
    m_slicing: Optional[str] = None,
) -> PlantSegImage:
    """
    Open an image file and create a PlantSegImage object.

    Args:
        path (Path): Path to the image file
        key (str): Key to load data from h5 or zarr files
        image_name (str): Name of the image (a unique name to identify the image)
        semantic_type (str): Semantic type of the image, should be raw, segmentation, prediction or label
        stack_layout (str): Layout of the image, should be XY, CXY, ZXY, CZXY or ZCXY
        channel (int): Channel to load from the image, should be an integer if the image is multichannel.
        m_slicing (str): Slicing to apply to the image, should be a string with the format [start:stop, ...] for each dimension.
    """
    data, voxel_size = _load_data(path, key)

    stack_layout = ImageLayout(stack_layout)

    if m_slicing is not None:
        data = dp.image_crop(data, m_slicing)

    data, stack_layout = _select_channel(data, channel, stack_layout)
    data = _add_singletons(data, stack_layout)

    image_properties = ImageProperties(
        name=image_name,
        semantic_type=SemanticType(semantic_type),
        voxel_size=voxel_size,
        image_layout=stack_layout,
        original_voxel_size=voxel_size,
    )

    return PlantSegImage(data=data, properties=image_properties)


def _image_postprocessing(image: PlantSegImage, scale_to_origin: bool, export_dtype) -> PlantSegImage:
    if scale_to_origin and image.requires_scaling:
        data = dp.scale_image_to_voxelsize(
            image.data,
            input_voxel_size=image.voxel_size,
            output_voxel_size=image.original_voxel_size,
            order=image.interpolation_order(),
        )
        new_voxel_size = image.original_voxel_size
    else:
        data = image.data
        new_voxel_size = image.voxel_size

    if image.image_type == ImageType.IMAGE:
        data = dp.normalize_01(data)
        if export_dtype in ["uint8", "uint16"]:
            max_val = np.iinfo(export_dtype).max
            data = (data * max_val).astype(export_dtype)
        elif export_dtype in ["float32", "float64"]:
            data = data.astype(export_dtype)

        else:
            raise ValueError(f"Data type {export_dtype} not recognized, should be uint8, uint16, float32 or float64")

    elif image.image_type == ImageType.LABEL:
        if export_dtype in ["float32", "float64"]:
            raise ValueError(f"Data type {export_dtype} not recognized for label image, should be uint8 or uint16")
        data = data.astype(export_dtype)

    else:
        raise ValueError(f"Image type {image.image_type} not recognized, should be image or label")

    return data, new_voxel_size


def save_image(
    image: PlantSegImage,
    directory: Path,
    file_name: str,
    custom_key: str,
    scale_to_origin: bool,
    file_format: str = "tiff",
    dtype: str = "uint16",
) -> None:
    """
    Write an PlantSegImage object to disk.

    Args:
        image (PlantSegImage): Image to save
        directory (Path): Directory to save the image
        file_name (str): Name of the file
        custom_key (str): Custom key to save the image (it will be used as suffix in tiff files or as key in h5 or zarr files)
        scale_to_origin (bool): Scale the image to the original voxel size (if different from the current voxel size)
        file_format (str): File format to save the image, should be tiff, h5 or zarr
        dtype (str): Data type to save the image, should be uint8, uint16, float32 or float64
    """
    data, voxel_size = _image_postprocessing(image, scale_to_origin, dtype)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if file_format == "tiff":
        file_path_name = directory / f"{file_name}_{custom_key}.tiff"
        create_tiff(file_path_name, data, voxel_size)

    elif file_format == "zarr":
        file_path_name = directory / f"{file_name}.zarr"
        create_zarr(
            path=file_path_name,
            data=data,
            voxel_size=voxel_size,
            key=custom_key,
        )

    elif file_format == "h5":
        file_path_name = directory / f"{file_name}.h5"
        create_h5(file_path_name, data, voxel_size, key=custom_key)

    else:
        raise ValueError(f"File format {file_format} not recognized, should be tiff, h5 or zarr")
