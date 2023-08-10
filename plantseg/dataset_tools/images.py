from pathlib import Path
from typing import Union

import h5py
import numpy as np

from plantseg.io.h5 import create_h5, load_h5


class GenericImage:
    dimensionality: str
    key: str
    num_channels: int
    data: Union[np.ndarray, h5py.Dataset]
    layout: str = 'xy'

    def __init__(self, data: np.ndarray,
                 key: str = 'raw',
                 dimensionality: str = '3D'):
        """
        Generic image class to handle 2D and 3D images consistently.
        Args:
            data (np.ndarray): image data
            key (str): key to use when saving to h5
            dimensionality (str): '2D' or '3D'
        """
        assert dimensionality in ['2D', '3D'], f'Invalid dimensionality: {dimensionality}, valid values are: 2D, 3D.'
        self.data = data
        self.key = key
        self.dimensionality = dimensionality
        self.num_channels = 1

    def __repr__(self):
        return (f'{self.__class__.__name__}(dimensionality={self.dimensionality},'
                f' shape={self.shape},'
                f' layout={self.layout})')

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def clean_shape(self) -> tuple[int, ...]:
        """
        Returns the shape without singleton dimensions and channel dimensions
        """
        assert len(self.shape) == len(self.layout), f'Shape and layout do not match: {self.shape} vs {self.layout}.'
        clean_shape = [s for s, l in zip(self.shape, self.layout) if 'c' != l and '1' != l]
        return tuple(clean_shape)

    def load_data(self):
        """
        Load the data from the h5 file
        """
        if isinstance(self.data, h5py.Dataset):
            self.data = self.data[...]

    def remove_singletons(self, remove_channel: bool = False):
        """
        Remove singleton dimensions from the data, and optionally remove the
            channel dimension if channel dimension is 1.
        Args:
            remove_channel (bool): if True, remove the channel dimension if it is 1.
        Returns:
            GenericImage: self

        """
        data = self.data
        axis_to_squeeze = []
        for i, l in enumerate(self.layout):
            if 'c' == l and remove_channel:
                axis_to_squeeze.append(i)
            if '1' == l:
                axis_to_squeeze.append(i)

        if axis_to_squeeze:
            data = np.squeeze(data, axis=tuple(axis_to_squeeze))
            return type(self)(data=data, key=self.key, dimensionality=self.dimensionality)

        return self

    @classmethod
    def from_h5(cls, path: Union[str, Path], key: str, dimensionality: str, load_data: bool = False):
        """
        Instantiate a GenericImage from a h5 file
        """
        if load_data:
            data, _ = load_h5(path=path, key=key)
        else:
            data = h5py.File(path, mode='r')[key]
        return cls(data=data, key=key, dimensionality=dimensionality)

    def to_h5(self, path: Union[str, Path],
              voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
              mode: str = 'a'):
        """
        Save the data to a h5 file
        Args:
            path (str, Path): path to the h5 file
            voxel_size (tuple): voxel size
            mode (str): 'a' to append to an existing file, 'w' to overwrite an existing file
        """
        self.load_data()
        create_h5(path, stack=self.data, key=self.key, voxel_size=voxel_size, mode=mode)


class Image(GenericImage):
    def __init__(self, data: np.ndarray,
                 key: str = 'raw',
                 dimensionality: str = '3D'):
        """
        Args:
            data: numpy array
            key: internal key of the dataset
            dimensionality: 2D or 3D
        """
        super().__init__(data=data, key=key, dimensionality=dimensionality)

        if dimensionality == '2D':
            if data.ndim == 2:
                self.num_channels = 1
                self.layout = 'xy'
            elif data.ndim == 3:
                self.num_channels = data.shape[0]
                self.layout = 'cxy'
            elif data.ndim == 4:
                self.num_channels = data.shape[0]
                assert data.shape[1] == 1, f'Invalid number of channels: {data.shape[1]}, expected 1.'
                self.layout = 'c1xy'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 2 or 3 or 4.')

        elif dimensionality == '3D':
            if data.ndim == 3:
                self.num_channels = 1
                self.layout = 'xyz'
            elif data.ndim == 4:
                self.num_channels = data.shape[0]
                self.layout = 'cxyz'
            raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 3 or 4.')

        else:
            raise ValueError(f'Invalid dimensionality: {dimensionality}, valid values are: 2D, 3D.')


class Labels(GenericImage):
    def __init__(self, data: np.ndarray,
                 key: str = 'labels',
                 dimensionality: str = '3D'):
        """
        Args:
            data: numpy array
            key: internal key of the dataset
            dimensionality: 2D or 3D
        """
        super().__init__(data=data, key=key, dimensionality=dimensionality)

        if dimensionality == '2D':
            if data.ndim == 2:
                self.num_channels = 1
                self.layout = 'xy'
            elif data.ndim == 3:
                assert data.shape[0] == 1, f'Invalid number of channels: {data.shape[0]}, expected 1.'
                self.layout = '1xy'
            elif data.ndim == 4:
                assert data.shape[0] == 1, f'Invalid number of channels: {data.shape[0]}, expected 1.'
                assert data.shape[1] == 1, f'Invalid number of channels: {data.shape[1]}, expected 1.'
                self.layout = '11xy'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 2 or 3 or 4.')

        elif dimensionality == '3D':
            if data.ndim == 3:
                self.num_channels = 1
                self.layout = 'xyz'
            elif data.ndim == 4:
                assert data.shape[0] == 1, f'Invalid number of channels: {data.shape[0]}, expected 1.'
                self.layout = '1xyz'

        else:
            raise ValueError(f'Invalid dimensionality: {dimensionality}, valid values are: 2D, 3D.')


class Stack:
    dimensionality: str
    layout: str
    data: {}

    def __init__(self, *images: GenericImage,
                 dimensionality: str = '3D',
                 strict: bool = True):
        """
        Args:
            *images (GenericImage): list of images
            dimensionality (str): 2D or 3D
            strict (bool): if True, raise an error if the images do not have the same dimensionality
        """
        self.dimensionality = dimensionality

        data = {}
        dimensionality = images[0].dimensionality
        for image in images:
            assert image.dimensionality == dimensionality, (f'Invalid dimensionality: {image.dimensionality},'
                                                            f' all images must have the same dimensionality.')
            data[image.key] = image

        self.data = data
        result, msg = self.validate()

        if not result and strict:
            raise ValueError(msg)

    @property
    def keys(self) -> list[str]:
        """
        list all the keys of the stack
        """
        return list(self.data.keys())

    @property
    def clean_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the stack without the channel dimension and singleton dimensions
        """
        key = self.keys[0]
        return self.data[key].clean_shape

    def validate_dimensionality(self) -> tuple[bool, str]:
        for image in self.data.values():
            if image.dimensionality != self.dimensionality:
                msg = (f'Invalid dimensionality: {image.dimensionality}, all'
                       f' images must have the same dimensionality.')
                return False, msg

        return True, ''

    def validate_layout(self) -> tuple[bool, str]:
        for type_image in [Image, Labels]:
            list_image = [image for image in self.data.values() if isinstance(image, type_image)]
            if list_image:
                layout = list_image[0].layout
                for image in list_image:
                    if image.layout != layout:
                        msg = (f'Invalid layout: {image.layout}, all'
                               f' images of type {self.__class__.__name__} must have the same layout.')
                        return False, msg
        return True, ''

    def validate_shape(self) -> tuple[bool, str]:
        if len(self.data) == 0:
            return False, 'Empty stack.'

        for image in self.data.values():
            if image.clean_shape != self.clean_shape:
                msg = (f'Invalid clean shape: {image.shape},'
                       f' all images must have the clean same'
                       f' shape {self.clean_shape}.')
                return False, msg
        return True, ''

    def validate(self) -> tuple[bool, str]:
        """
        Validate the stack to ensure that all images have the same dimensionality, layout and shape.
        """
        for test in [self.validate_dimensionality, self.validate_layout, self.validate_shape]:
            result, msg = test()
            if not result:
                return False, msg
        return True, ''

    def dump_to_h5(self, path: Union[str, Path], mode: str = 'a'):
        """
        Dump the full stack to an HDF5 file.
        Args:
            path: path to the HDF5 file
            mode: write mode, one of ['w', 'a', 'r+', 'w-']

        """
        assert mode in ['w', 'a', 'r+', 'w-'], f'Invalid mode: {mode}, valid values are: [w, a, r+, w-].'
        for key, stack in self.data.items():
            stack.to_h5(path=path, mode=mode)
            # switch to append mode after first iteration
            mode = 'a'

    @classmethod
    def from_h5(cls, path: Union[str, Path],
                keys: tuple[tuple[str, GenericImage]],
                dimensionality: str,
                load_data: bool = False,
                strict: bool = True):
        """
        Load the full stack from an HDF5 file.
        Args:
            path: path to the HDF5 file
            keys: list of (keys, type of data) to load
            dimensionality: 2D or 3D
            load_data: if True, load the data from the HDF5 file
            strict: if True, raise an error if the images do not have the same dimensionality
        """
        data = []
        for key, type_image in keys:
            im = type_image.from_h5(path=path, key=key, dimensionality=dimensionality, load_data=load_data)
            data.append(im)

        return cls(*data, dimensionality=dimensionality, strict=strict)
