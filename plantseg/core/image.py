import logging
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from pydantic import BaseModel

import plantseg.functionals.dataprocessing as dp
from plantseg.core.voxelsize import VoxelSize
from plantseg.io import (
    H5_EXTENSIONS,
    PIL_EXTENSIONS,
    TIFF_EXTENSIONS,
    create_h5,
    create_tiff,
    create_zarr,
    load_h5,
    load_pil,
    load_tiff,
    load_zarr,
    read_h5_voxel_size,
    read_tiff_voxel_size,
    read_zarr_voxel_size,
)

logger = logging.getLogger(__name__)


class SemanticType(Enum):
    """
    Enum class for image types.

    Attributes:
        RAW (str): Reserved for raw images (e.g. microscopy images)
        LABEL (str): Reserved for ground truth labels
        PREDICTION (str): Reserved for model prediction
        SEGMENTATION (str): Reserved for segmentation masks
    """

    RAW = "raw"
    LABEL = "label"
    PREDICTION = "prediction"
    SEGMENTATION = "segmentation"


class ImageType(Enum):
    """
    Enum class for image types.

    Attributes:
        IMAGE (str): Image data
        LABEL (str): Label data
    """

    IMAGE = "image"
    LABEL = "labels"

    @classmethod
    def to_choices(cls) -> list[str]:
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
    """
    Enum class for image layout.
    Axis available are:
    - X: Width
    - Y: Height
    - Z: Depth
    - C: Channel

    Attributes:
        YX (str): 2D image with X and Y axis
        CYX (str): 2D image with Channel, X and Y axis
        ZYX (str): 3D image with Z, X and Y axis
        CZYX (str): 3D image with Channel, Z, X and Y axis
        ZCYX (str): 3D image with Z, Channel, X and
    """

    YX = "YX"
    CYX = "CYX"
    ZYX = "ZYX"
    CZYX = "CZYX"
    ZCYX = "ZCYX"  # This is not supported, should be converted to CZYX during import

    @classmethod
    def to_choices(cls) -> list[str]:
        return [il.value for il in cls]


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

    name: str
    semantic_type: SemanticType
    voxel_size: VoxelSize
    image_layout: ImageLayout
    original_voxel_size: VoxelSize

    @property
    def dimensionality(self) -> ImageDimensionality:
        if self.image_layout in (ImageLayout.YX, ImageLayout.CYX):
            return ImageDimensionality.TWO
        elif self.image_layout in (
            ImageLayout.ZYX,
            ImageLayout.CZYX,
            ImageLayout.ZCYX,
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
    def channel_axis(self) -> int | None:
        if self.image_layout in (ImageLayout.CYX, ImageLayout.CZYX):
            return 0
        elif self.image_layout == ImageLayout.ZCYX:
            return 1
        elif self.image_layout in (ImageLayout.YX, ImageLayout.ZYX):
            return None
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

    def interpolation_order(self, image_default: int = 1) -> int:
        if self.image_type == ImageType.LABEL:
            return 0
        elif self.image_type == ImageType.IMAGE:
            return image_default
        else:
            raise ValueError(f"Image type {self.image_type} not recognized")


def scale_to_voxelsize(scale: tuple[float, ...], layout: ImageLayout, unit: str = 'um') -> VoxelSize:
    if layout == ImageLayout.YX:
        assert len(scale) == 2, f"Scale should have 2 elements for layout {layout}"
        return VoxelSize(voxels_size=(1.0, scale[0], scale[1]), unit=unit)

    elif layout == ImageLayout.CYX:
        assert len(scale) == 3, f"Scale should have 3 elements for layout {layout}"
        return VoxelSize(voxels_size=(1.0, scale[1], scale[2]), unit=unit)

    elif layout == ImageLayout.ZYX:
        assert len(scale) == 3, f"Scale should have 3 elements for layout {layout}"
        return VoxelSize(voxels_size=scale, unit=unit)

    elif layout == ImageLayout.CZYX:
        assert len(scale) == 4, f"Scale should have 4 elements for layout {layout}"
        return VoxelSize(voxels_size=(scale[1], scale[2], scale[3]), unit=unit)

    elif layout == ImageLayout.ZCYX:
        raise ValueError(f"Image layout {layout} not supported, should have been converted to CZYX")

    else:
        raise ValueError(f"Image layout {layout} not recognized")


class PlantSegImage:
    """Image class represents an image with its metadata and data."""

    _data: np.ndarray
    _properties: ImageProperties

    def __init__(self, data: np.ndarray, properties: ImageProperties):
        self._properties = properties
        data, properties = self._check_shape(data, properties)
        data = self._check_ndim(data)

        self._data = data
        self._properties = properties

        self._check_labels_have_no_channels()
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

        property_dict = self._properties.model_dump(exclude_none=True)

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

        if "original_voxel_size" not in metadata:
            raise ValueError("Original voxel size not found in metadata")

        original_voxel_size = VoxelSize(**metadata["original_voxel_size"])

        if "image_layout" not in metadata:
            raise ValueError("Image layout not found in metadata")

        image_layout = ImageLayout(metadata["image_layout"])

        if "voxel_size" not in metadata:
            raise ValueError("Voxel size not found in metadata")
        new_voxel_size = VoxelSize(**metadata["voxel_size"])

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
        Prepare and normalise the image to be loaded as a napari layer.

        All the metadata will be stored in the metadata of the layer.

        Returns:
            LayerDataTuple: Tuple containing the 0-1-normalised data, metadata, and type of the image.
        """
        # Dump the model properties to a dictionary
        metadata = self._properties.model_dump()

        # Preserve the ID in the metadata
        metadata["id"] = self.id

        # Create the LayerDataTuple
        layer_data_tuple = (
            self.get_data(),
            {
                "name": self.name,
                "scale": self.scale,
                "metadata": metadata,
            },
            self.image_type.value,
        )

        return LayerDataTuple(layer_data_tuple)

    def _check_ndim(self, data: np.ndarray) -> None:
        if self.image_layout in (ImageLayout.CYX, ImageLayout.ZYX):
            if data.ndim != 3:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 3 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.CZYX, ImageLayout.ZCYX):
            if data.ndim != 4:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 4 dimensions for layout {self.image_layout}"
                )

        elif self.image_layout in (ImageLayout.YX,):
            if data.ndim != 2:
                raise ValueError(
                    f"Data has shape {data.shape} but should have 2 dimensions for layout {self.image_layout}"
                )

        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

        return data

    def _check_shape(self, data: np.ndarray, properties: ImageProperties) -> None:
        if self.image_layout == ImageLayout.ZYX:
            if data.shape[0] == 1:
                logger.warning("Image layout is ZYX but data has only one z slice, casting to YX")
                properties.image_layout = ImageLayout.YX
                return data[0], properties

        elif self.image_layout == ImageLayout.CZYX:
            if data.shape[0] == 1 and data.shape[1] == 1:
                logger.warning("Image layout is CZYX but data has only one z slice and one channel, casting to YX")
                properties.image_layout = ImageLayout.YX
                return data[0, 0], properties

            elif data.shape[0] == 1 and data.shape[1] > 1:
                logger.warning("Image layout is CZYX but data has only one channel, casting to ZYX")
                properties.image_layout = ImageLayout.ZYX
                return data[0], properties

            elif data.shape[0] > 1 and data.shape[1] == 1:
                logger.warning("Image layout is CZYX but data has only one z slice, casting to CYX")
                properties.image_layout = ImageLayout.CYX
                return data[:, 0], properties.image_layout

        elif self.image_layout == ImageLayout.ZCYX:
            raise ValueError(f"Image layout {self.image_layout} not supported, should have been converted to CZYX")

        return data, properties

    def _check_labels_have_no_channels(self) -> None:
        if self.image_type == ImageType.LABEL:
            if self.channel_axis is not None:
                raise ValueError(f"Label images should not have channel axis, but found layout {self.image_layout}")

    @property
    def requires_scaling(self) -> bool:
        """Returns True if the image has a different voxel size."""
        if self.voxel_size != self.original_voxel_size:
            return True
        return False

    def _get_data_channel_layout(self, channel: int | None = None, normalize_01: bool = True) -> np.ndarray:
        """Get the data if the layout is multichannel."""
        if channel is None:
            data = self._data
            if normalize_01:
                data = dp.normalize_01_channel_wise(data, self.channel_axis)
            return data

        if channel < 0:
            raise ValueError(f"Channel should be a positive integer, but got {channel}")

        if channel > self._data.shape[self.channel_axis]:
            raise ValueError(
                f"Channel {channel} is out of bounds, the image has {self._data.shape[self.channel_axis]} channels"
            )

        data = dp.select_channel(self._data, channel, self.channel_axis)
        if normalize_01:
            data = dp.normalize_01(data)
        return data

    def _get_data(self, normalize_01: bool = True) -> np.ndarray:
        """Get the data if the layout is not multichannel."""
        data = self._data
        if normalize_01:
            data = dp.normalize_01(data)
        return data

    def get_data(self, channel: int | None = None, normalize_01: bool = True) -> np.ndarray:
        """Returns the data of the image.

        Args:
            channel (int): Channel to load from the image (if the image is multichannel). If None, all channels are loaded.
            normalize_01 (bool): Normalize the data between 0 and 1, if the image is a Label image, the data is not normalized.
        """
        if self.image_type == ImageType.LABEL:
            return self._data

        if self.channel_axis is not None:
            return self._get_data_channel_layout(channel, normalize_01)

        return self._get_data(normalize_01)

    @property
    def scale(self) -> tuple[float, ...]:
        """Returns the scale of the image.

        The scale is equal to the voxel size in each spatial dimension and 1 in other channels.
        """
        if self.image_layout == ImageLayout.YX:
            return (self.voxel_size.x, self.voxel_size.y)
        elif self.image_layout == ImageLayout.ZYX:
            return (self.voxel_size.z, self.voxel_size.x, self.voxel_size.y)
        elif self.image_layout == ImageLayout.CYX:
            return (1.0, self.voxel_size.x, self.voxel_size.y)
        elif self.image_layout == ImageLayout.CZYX:
            return (1.0, self.voxel_size.z, self.voxel_size.x, self.voxel_size.y)
        elif self.image_layout == ImageLayout.ZCYX:
            raise ValueError(f"Image layout {self.image_layout} not supported, should have been converted to CZYX")
        else:
            raise ValueError(f"Image layout {self.image_layout} not recognized")

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the image."""
        return self._data.shape

    @property
    def properties(self) -> ImageProperties:
        """Returns the properties of the image."""
        return self._properties

    @property
    def voxel_size(self) -> VoxelSize:
        """Returns the voxel size of the image."""
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
    def channel_axis(self) -> int | None:
        return self._properties.channel_axis

    @property
    def is_multichannel(self) -> bool:
        """Returns True if the image is multichannel, False otherwise."""
        return self.channel_axis is not None

    def interpolation_order(self, image_default: int = 1) -> int:
        """Returns the default interpolation order used for the image."""
        return self._properties.interpolation_order(image_default)

    def has_valid_voxel_size(self) -> bool:
        """Returns True if the voxel size is valid (not None), False otherwise."""
        return self.voxel_size.voxels_size is not None

    def has_valid_original_voxel_size(self) -> bool:
        """Returns True if the original voxel size is valid (not None), False otherwise."""
        return self.original_voxel_size.voxels_size is not None


def _load_data(path: Path, key: str | None) -> tuple[np.ndarray, VoxelSize]:
    """Load data and voxel size from a file."""
    ext = path.suffix
    if key == "":
        key = None

    if ext in H5_EXTENSIONS:
        return load_h5(path, key), read_h5_voxel_size(path, key)

    if ext in TIFF_EXTENSIONS:
        return load_tiff(path), read_tiff_voxel_size(path)

    if ext in PIL_EXTENSIONS:
        return load_pil(path), VoxelSize()

    if ".zarr" in path.suffixes:
        return load_zarr(path, key), read_zarr_voxel_size(path, key)

    raise NotImplementedError(f"File extension '{ext}' is not supported for path: {path}")


def import_image(
    path: Path,
    key: str | None = None,
    image_name: str = "image",
    semantic_type: str = "raw",
    stack_layout: str = "YX",
    m_slicing: str | None = None,
) -> PlantSegImage:
    """
    Open an image file and create a PlantSegImage object.

    Args:
        path (Path): Path to the image file
        key (str): Key to load data from h5 or zarr files
        image_name (str): Name of the image (a unique name to identify the image)
        semantic_type (str): Semantic type of the image, should be raw, segmentation, prediction or label
        stack_layout (str): Layout of the image, should be YX, CYX, ZYX, CZYX or ZCYX
        m_slicing (str): Slicing to apply to the image, should be a string with the format [start:stop, ...] for each dimension.
    """
    data, voxel_size = _load_data(path, key)

    image_layout = ImageLayout(stack_layout)
    if image_layout is ImageLayout.ZCYX:  # then make it CZYX
        data = np.moveaxis(data, 0, 1)
        image_layout = ImageLayout.CZYX

    if m_slicing is not None:
        data = dp.image_crop(data, m_slicing)

    image_properties = ImageProperties(
        name=image_name,
        semantic_type=SemanticType(semantic_type),
        voxel_size=voxel_size,
        image_layout=image_layout,
        original_voxel_size=voxel_size,
    )

    return PlantSegImage(data=data, properties=image_properties)


def _image_postprocessing(
    image: PlantSegImage, scale_to_origin: bool, export_dtype: str
) -> tuple[np.ndarray, VoxelSize]:
    if scale_to_origin and image.requires_scaling:
        data = dp.scale_image_to_voxelsize(
            image.get_data(),
            input_voxel_size=image.voxel_size,
            output_voxel_size=image.original_voxel_size,
            order=image.interpolation_order(),
        )
        new_voxel_size = image.original_voxel_size
    else:
        data = image.get_data()
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
    Write a PlantSegImage object to disk.

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
        create_tiff(path=file_path_name, stack=data, voxel_size=voxel_size, layout=image.image_layout.value)

    elif file_format == "zarr":
        file_path_name = directory / f"{file_name}.zarr"
        create_zarr(
            path=file_path_name,
            stack=data,
            voxel_size=voxel_size,
            key=custom_key,
        )

    elif file_format == "h5":
        file_path_name = directory / f"{file_name}.h5"
        create_h5(path=file_path_name, stack=data, voxel_size=voxel_size, key=custom_key)

    else:
        raise ValueError(f"File format {file_format} not recognized, should be tiff, h5 or zarr")
