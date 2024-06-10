from typing import Protocol, Optional
import numpy as np
from pydantic import BaseModel
from enum import Enum
from io.utils import VoxelSize, DataHandler


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
    XY = ("XY", ImageDimensionality.TWO)
    CXY = ("CXY", ImageDimensionality.TWO)
    ZXY = ("ZXY", ImageDimensionality.THREE)
    CZXY = ("CZXY", ImageDimensionality.THREE)
    ZCXY = ("ZCXY", ImageDimensionality.THREE)

    def __init__(self, layout: str, dimensionality: ImageDimensionality):
        self.layout = layout
        self.dimensionality = dimensionality


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


class NumpyDataHandler:
    """
    In-memory data handler for numpy arrays.
    """

    def __init__(self, data: np.ndarray, voxel_size: VoxelSize) -> None:
        self._data = data
        self._voxel_size = voxel_size

    def get_data(self) -> np.ndarray:
        return self._data

    def get_shape(self) -> tuple[int, ...]:
        return self._data.shape

    def get_voxel_size(self) -> VoxelSize:
        return self._voxel_size


class Image:
    _data: DataHandler
    _image_properties: ImageProperties

    def __init__(self, image_properties: ImageProperties, data: DataHandler) -> None:
        self._data = data
        self._image_properties = image_properties

        self._check_dimensionality()

    def _check_dimensionality(self) -> None:
        # Validate that the data shape matches the image layout
        shape = self._data.get_shape()
        if self._image_properties.image_layout.dimensionality == ImageDimensionality.TWO:
            assert len(shape) == 2 or (len(shape) == 3 and shape[0] == 1)

        elif self._image_properties.image_layout.dimensionality == ImageDimensionality.THREE:
            if self._image_properties.image_layout == ImageLayout.ZXY:
                assert len(shape) == 3
            elif self._image_properties.image_layout == ImageLayout.CZXY:
                assert len(shape) == 4 and shape[0] == 1
            elif self._image_properties.image_layout == ImageLayout.ZCXY:
                assert len(shape) == 4 and shape[1] == 1

        else:
            raise ValueError(f"Invalid dimensionality: {self._image_properties.image_layout.dimensionality}")

    @property
    def data_handler(self) -> DataHandler:
        return self._data

    @property
    def properties(self) -> ImageProperties:
        return self._image_properties
