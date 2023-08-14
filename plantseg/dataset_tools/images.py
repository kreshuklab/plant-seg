from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from plantseg.io.h5 import create_h5, load_h5, read_attribute_h5, write_attribute_h5
from plantseg.io.h5 import list_keys as list_keys_h5


class MockData:
    ndim: int
    shape: tuple[int]
    key: str
    path: Path

    def __init__(self, path: Path, key: str):
        assert path.exists(), f'Path does not exist: {path}'
        with h5py.File(path, mode='r') as f:
            assert key in f, f'Key not found in file: {key}'
            data = f[key]
            self.ndim = data.ndim
            self.shape = data.shape

        self.key = key
        self.path = path

    def load(self):
        data, infos = load_h5(self.path, key=self.key)
        return data, infos


@dataclass
class ImageSpecs:
    key: str
    data_type: str
    dimensionality: str
    num_channels: int = 1
    is_sparse: bool = None

    def __post_init__(self):
        assert self.data_type in ('image', 'labels'), f'data_type must be either image or label, found {self.data_type}'
        assert self.dimensionality in ('2D', '3D'), (f'dimensionality must be either'
                                                     f' 2D or 3D, found {self.dimensionality}')
        assert isinstance(self.num_channels, int), f'num_channels must be an integer, found {self.num_channels}'
        assert self.num_channels > 0, f'num_channels must be greater than 0, found {self.num_channels}'
        assert isinstance(self.key, str), f'key must be a string, found {self.key}'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_h5(cls, path: Union[str, Path], key: str):
        attrs = read_attribute_h5(path=path, key=key)
        attrs = {k: v for k, v in attrs.items() if k in cls.__annotations__.keys()}
        attrs['num_channels'] = int(attrs['num_channels'])

        if attrs['is_sparse'] is not None:
            attrs['is_sparse'] = bool(attrs['is_sparse'])
        return cls(**attrs)


@dataclass
class StackSpecs:
    dimensionality: str = '3D'
    list_specs: list[ImageSpecs] = field(default_factory=list)

    def __post_init__(self):
        assert self.dimensionality in ('2D', '3D'), (f'dimensionality must be either'
                                                     f' 2D or 3D, found {self.dimensionality}')
        for image in self.list_specs:
            assert isinstance(image, ImageSpecs), f'list_images must contain ImageSpec objects, found {image}'

    @classmethod
    def from_dict(cls, dict_specs):
        list_specs = dict_specs.pop('list_specs')
        list_specs = [ImageSpecs(**spec) for spec in list_specs]
        return cls(list_specs=list_specs, **dict_specs)

    def to_dict(self):
        return asdict(self)


class GenericImage:
    dimensionality: str
    key: str
    num_channels: int
    data: Union[np.ndarray, MockData]
    layout: str = 'xy'
    is_sparse: bool
    data_type: str
    infos: tuple = None

    def __init__(self, data: np.ndarray,
                 spec: ImageSpecs):
        """
        Generic image class to handle 2D and 3D images consistently.
        Args:
            data (np.ndarray): image data
            spec (ImageSpecs): image specifications template to be used to create the image
        """
        self.data = data
        self.spec = spec

    def __repr__(self):
        return (f'{self.__class__.__name__}(dimensionality={self.dimensionality},'
                f' shape={self.shape},'
                f' layout={self.layout})')

    @property
    def key(self):
        return self.spec.key

    @property
    def is_sparse(self):
        return self.spec.is_sparse

    @property
    def data_type(self):
        return self.spec.data_type

    @property
    def dimensionality(self):
        return self.spec.dimensionality

    @property
    def num_channels(self):
        return self.spec.num_channels

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
        clean_shape = [s for s, l in zip(self.shape, self.layout) if 'c' != l and '1' != l]
        return tuple(clean_shape)

    def load_data(self) -> np.ndarray:
        """
        Load the data from the h5 file
        """
        if isinstance(self.data, MockData):
            data, infos = self.data.load()
            self.infos = infos
            return data

        return self.data

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
            spec = ImageSpecs(key=self.key,
                              data_type=self.data_type,
                              dimensionality=self.dimensionality,
                              num_channels=self.num_channels,
                              is_sparse=self.is_sparse)
            return type(self)(data=data, spec=spec)

        return self

    def check_compatibility(self, spec: ImageSpecs) -> tuple[bool, str]:
        """
        Validate the image against a specification
        Args:
            spec (ImageSpecs): specification to validate against
        Returns:
            bool: True if the image is valid, False otherwise
            str: error message
        """
        if self.data_type != spec.data_type:
            return False, f'Invalid data type: {self.data_type} vs {spec.data_type}'

        if self.dimensionality != spec.dimensionality:
            return False, f'Invalid dimensionality: {self.dimensionality} vs {spec.dimensionality}'

        if self.num_channels != spec.num_channels:
            return False, f'Invalid number of channels: {self.num_channels} vs {spec.num_channels}'

        if self.data_type == 'labels' and self.is_sparse != spec.is_sparse:
            return False, f'Invalid sparsity: {self.is_sparse} vs {spec.is_sparse}'

        return True, ''

    @classmethod
    def from_h5(cls, path: Union[str, Path], key: str = None, spec: ImageSpecs = None):
        """
        Instantiate a GenericImage from a h5 file
        Args:
            path (str, Path): path to the h5 file
            key (str): key of the image in the h5 file
            spec (ImageSpecs): image specifications template to be used to create the image
        Returns:
            GenericImage: instance of GenericImage
        """
        assert key is not None or spec is not None, 'Either key or spec must be provided.'

        if spec is None:
            spec = ImageSpecs.from_h5(path=path, key=key)

        data = MockData(path=path, key=spec.key)
        return cls(data=data, spec=spec)

    def to_h5(self, path: Union[str, Path],
              mode: str = 'a'):
        """
        Save the data to a h5 file
        Args:
            path (str, Path): path to the h5 file
            mode (str): 'a' to append to an existing file, 'w' to overwrite an existing file
        """
        data = self.load_data()
        create_h5(path, stack=data, key=self.key, mode=mode)
        write_attribute_h5(path=path, key=self.key, atr_dict=self.spec.to_dict())


class Image(GenericImage):
    def __init__(self, data: np.ndarray,
                 spec: ImageSpecs):
        """
        Args:
            data: numpy array
            spec: ImageSpecs containing the specifications of the image
        """
        assert spec.data_type == 'image', f'Invalid data type: {spec.data_type}, expected image.'
        assert spec.is_sparse is None, f'Invalid sparsity: {spec.is_sparse}, expected None for images.'
        dimensionality = spec.dimensionality

        if dimensionality == '2D':
            if data.ndim == 2:
                num_channels = 1
                layout = 'xy'
            elif data.ndim == 3:
                num_channels = data.shape[0]
                assert num_channels == spec.num_channels, (f'Invalid shape for 2D image: expected number of channels '
                                                           f'{spec.num_channels}, got {num_channels}')
                layout = 'cxy'
            elif data.ndim == 4:
                num_channels = data.shape[0]
                assert num_channels == spec.num_channels, (f'Invalid shape for 2D image: expected number of channels '
                                                              f'{spec.num_channels}, got {num_channels}')
                assert data.shape[1] == 1, (f'Invalid shape for 2D image: {data.shape}, expected (C, 1, X, Y), '
                                            f'got (c, {data.shape[1]}, x, y')
                layout = 'c1xy'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 2 or 3 or 4.')

        elif dimensionality == '3D':
            if data.ndim == 3:
                num_channels = 1
                layout = 'xyz'
            elif data.ndim == 4:
                num_channels = data.shape[0]
                assert num_channels == spec.num_channels, (f'Invalid shape for 3D image: expected number of channels '
                                                              f'{spec.num_channels}, got {num_channels}')
                layout = 'cxyz'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 3 or 4.')

        else:
            raise ValueError(f'Invalid dimensionality: {dimensionality}, valid values are: 2D, 3D.')

        spec.num_channels = num_channels
        spec.layout = layout
        super().__init__(data=data, spec=spec)


class Labels(GenericImage):
    def __init__(self, data: np.ndarray,
                 spec: ImageSpecs):
        """
        Args:
            data: numpy array
            spec: ImageSpecs containing the specifications of a label image
        """
        assert spec.data_type == 'labels', f'Invalid data type: {spec.data_type}, expected labels.'
        assert spec.is_sparse is not None, 'Sparse flag must be set for labels.'
        assert spec.num_channels == 1, f'Invalid number of channels: {spec.num_channels}, expected 1.'

        dimensionality = spec.dimensionality
        if dimensionality == '2D':
            if data.ndim == 2:
                layout = 'xy'
            elif data.ndim == 3:
                assert data.shape[0] == 1, (f'Invalid shape for 2D labels. '
                                            f'Expected shape: (1, y, x) got ({data.shape[0]}, y, x).')
                layout = '1xy'
            elif data.ndim == 4:
                assert data.shape[0] == 1 and data.shape[1], (f'Invalid shape for 2D labels. '
                                                              f'Expected shape: (1, 1, y, x) got'
                                                              f' ({data.shape[0]}, {data.shape[1]} y, x).')
                layout = '11xy'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 2 or 3 or 4.')

        elif dimensionality == '3D':
            if data.ndim == 3:
                layout = 'xyz'
            elif data.ndim == 4:
                assert data.shape[0] == 1, (f'Invalid shape for 3D labels. '
                                            f'Expected shape: (1, z, y, x) got'
                                            f' ({data.shape[0]}, z, y, x).')
                layout = '1xyz'
            else:
                raise ValueError(f'Invalid number of dimensions: {data.ndim}, expected 3 or 4.')

        else:
            raise ValueError(f'Invalid dimensionality: {dimensionality}, valid values are: 2D, 3D.')

        spec.layout = layout
        super().__init__(data=data, spec=spec)


available_image_types = {'image': Image, 'labels': Labels}


def from_spec(data: np.ndarray, spec: ImageSpecs) -> GenericImage:
    """
    Create an image from a specification
    Args:
        data: numpy array
        spec: ImageSpecs containing the specifications of the image
    Returns:
        GenericImage: image
    """
    data_type = spec.data_type
    if data_type not in available_image_types:
        raise ValueError(f'Invalid image type: {data_type}, valid values are: {available_image_types.keys()}')
    return available_image_types[data_type](data=data, spec=spec)


def from_h5(path: Union[str, Path], key: str) -> GenericImage:
    """
    Load an image from a h5 file
    Args:
        path (str, Path): path to the h5 file
        key (str): key of the image in the h5 file
    Returns:
        GenericImage: image
    """
    spec = ImageSpecs.from_h5(path=path, key=key)
    image_type = available_image_types[spec.data_type]
    data = image_type.from_h5(path=path, key=key)
    return data


class Stack:
    dimensionality: str
    list_specs: list[ImageSpecs]
    data: dict[str, GenericImage] = {}

    def __init__(self, *images: Union[GenericImage, np.ndarray],
                 spec: StackSpecs,
                 strict: bool = True):
        """
        Args:
            *images (GenericImage): list of images
            spec (StackSpecs): specification of the stack
            strict (bool): if True, raise an error if the stack is invalid
        """
        self.spec = spec
        assert len(images) == len(self.list_specs), (f'Invalid number of images: {len(images)}, '
                                                     f'expected {len(self.list_specs)}.')
        data = {}
        for image, spec in zip(images, self.list_specs):
            if isinstance(image, np.ndarray):
                image = from_spec(data=image, spec=spec)
            data[spec.key] = image

        self.data = data
        result, msg = self.validate()

        if not result and strict:
            raise ValueError(msg)

    @property
    def dimensionality(self) -> str:
        """
        Return the dimensionality of the stack
        """
        return self.spec.dimensionality

    @property
    def list_specs(self) -> list[ImageSpecs]:
        """
        Return the list of image specifications
        """
        return self.spec.list_specs

    @property
    def is_sparse(self) -> bool:
        """
        Check if the stack is sparse
        """
        labels_is_sparse = [spec.is_sparse for spec in self.list_specs if spec.data_type == 'labels']
        if len(labels_is_sparse) == 0:
            return False
        return all(labels_is_sparse)

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
        if len(self.keys) == 0:
            return ()
        key = self.keys[0]
        return self.data[key].clean_shape

    def validate_images(self, list_specs: list[ImageSpecs] = None) -> tuple[bool, str]:
        if list_specs is None:
            list_specs = self.list_specs

        for image, spec in zip(self.data.values(), list_specs):
            result, msg = image.check_compatibility(spec)
            if not result:
                return False, f'Error in {image.key}: {msg}'

        return True, ''

    def validate_dimensionality(self) -> tuple[bool, str]:
        for image in self.data.values():
            if image.dimensionality != self.dimensionality:
                msg = (f'Invalid dimensionality: {image.dimensionality}, all'
                       f' images must have the same dimensionality.')
                return False, msg

        return True, ''

    def validate_layout(self) -> tuple[bool, str]:
        # check image layout
        images_layout, labels_layout = [], []
        for image, spec in zip(self.data.values(), self.list_specs):
            if spec.data_type == 'image':
                images_layout.append(image.layout)
            elif spec.data_type == 'labels':
                labels_layout.append(image.layout)
            else:
                raise ValueError(f'Invalid image type: {spec.data_type}, valid values are: image, labels.')

        if len(images_layout) > 0:
            for layout in images_layout:
                if layout != images_layout[0]:
                    msg = f'Invalid layout found in images: {images_layout}'
                    return False, msg

        if len(labels_layout) > 0:
            for layout in labels_layout:
                if layout != labels_layout[0]:
                    msg = f'Invalid layout found in labels: {labels_layout}'
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
        for test in [self.validate_images,
                     self.validate_dimensionality,
                     self.validate_layout,
                     self.validate_shape]:
            result, msg = test()
            if not result:
                return False, msg
        return True, ''

    def check_compatibility(self, stack_spec: StackSpecs) -> tuple[bool, str]:

        if self.dimensionality != stack_spec.dimensionality:
            msg = (f'Invalid dimensionality: {self.dimensionality},'
                   f' expected {stack_spec.dimensionality}.')
            return False, msg

        if len(self.list_specs) != len(stack_spec.list_specs):
            msg = (f'Invalid number of images: {len(self.list_specs)},'
                   f' expected {len(stack_spec.list_specs)}.')
            return False, msg

        result, msg = self.validate_images(stack_spec.list_specs)
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

        keys, image_types = [], []
        for key, stack in self.data.items():
            stack.to_h5(path=path, mode=mode)
            # switch to append mode after first iteration
            mode = 'a'
            keys.append(key)

            for _name, _image_type in available_image_types.items():
                if isinstance(stack, _image_type):
                    image_types.append(_name)

        specs_dict = self.spec.to_dict()
        specs_dict.pop('list_specs')
        write_attribute_h5(path, atr_dict=specs_dict, key=None)

    @classmethod
    def from_h5(cls, path: Union[str, Path],
                expected_stack_specs: StackSpecs = None,
                strict: bool = True):
        """
        Load the full stack from an HDF5 file.
        Args:
            path: path to the HDF5 file
            expected_stack_specs: stack specifications
            strict: if True, raise an error if the images do not have the same dimensionality
        """
        if expected_stack_specs is None:
            stack_attrs = read_attribute_h5(path, key=None)
            stack_attrs = {k: v for k, v in stack_attrs.items() if k in StackSpecs.__annotations__.keys()}
            stack_attrs['list_specs'] = []
            stack_spec = StackSpecs.from_dict(stack_attrs)
            list_keys = list_keys_h5(path)
        else:
            stack_spec = expected_stack_specs
            list_keys = [s.key for s in stack_spec.list_specs]

        list_data = []
        list_specs_found = []
        for key in list_keys:
            data = from_h5(path, key=key)
            list_data.append(data)
            list_specs_found.append(data.spec)

        stack_spec.list_specs = list_specs_found
        stack = cls(*list_data, spec=stack_spec)

        if strict:
            result, msg = stack.validate()
            if not result:
                return stack, result, msg

        if expected_stack_specs is not None:
            result, msg = stack.check_compatibility(expected_stack_specs)
            if not result:
                return stack, result, msg

        return stack, True, ''
