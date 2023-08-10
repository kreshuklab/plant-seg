from pathlib import Path
from typing import Union

from plantseg.dataset_tools.dataset_handler import DatasetHandler
from plantseg.io.h5 import list_keys


class CheckDatasetDirectoryStructure:

    def __call__(self, dataset: DatasetHandler) -> tuple[bool, str]:
        # Check if dataset directory exists
        if not dataset.dataset_dir.exists():
            return False, f'Dataset directory {dataset.dataset_dir} does not exist.'

        # Check if dataset directory contains all expected subdirectories
        for phase in dataset.default_phases:
            if not (dataset.dataset_dir / phase).exists():
                return False, f'Dataset directory {dataset.dataset_dir} does not contain {phase} directory.'

        return True, ''


class CheckH5Keys:
    def __init__(self, expected_h5_keys: tuple[str, ...] = ('raw', 'labels')):
        self.expected_h5_keys = expected_h5_keys

    def __call__(self, stack: Union[str, Path]) -> tuple[bool, str]:
        found_keys = list_keys(stack)
        for key in self.expected_h5_keys:
            if key not in found_keys:
                return False, f'Key {key} not found in {stack}. Expected keys: {self.expected_h5_keys}'

        return True, ''


class CheckH5shapes:
    def __init__(self, dimensionality: str = '3D',
                 expected_h5_keys: tuple[str, ...] = (('raw', 'image'),
                                                      ('labels', 'labels')
                                                      )):
        """
        Check if the shape of the data in the h5 file matches the expected shape.
        Args:
            dimensionality: '2D' or '3D'
            expected_h5_keys: tuple of tuples, each tuple contains the key and the expected type of data
            possible types are: 'image', 'labels'
        """
        assert dimensionality in ['2D', '3D'], f'Invalid dimensionality: {dimensionality}, ' \
                                               f'valid values are: 2D, 3D'

        self.expected_shapes = {}
        if dimensionality == '2D':
            for key, data_type in expected_h5_keys:
                assert data_type in ['image', 'labels'], f'Invalid data type: {data_type}, ' \
                                                         f'valid values are: image, labels'
                if data_type == 'image':
                    self.expected_shapes[key] = [{'ndim': 2, 'shape': 'xy'},
                                                 {'ndim': 3, 'shape': 'cxy'},
                                                 {'ndim': 4, 'shape': 'c1xy'},
                                                 {'ndim': 4, 'shape': '1xy'}]
                elif data_type == 'labels':
                    self.expected_shapes[key] = [{'ndim': 2, 'shape': 'xy'},
                                                 {'ndim': 3, 'shape': '1xy'}]
        elif dimensionality == '3D':
            for key, data_type in expected_h5_keys:
                assert data_type in ['image', 'labels'], f'Invalid data type: {data_type}, ' \
                                                         f'valid values are: image, labels'
                if data_type == 'image':
                    self.expected_shapes[key] = [{'ndim': 3, 'shape': 'zxy'},
                                                 {'ndim': 4, 'shape': 'czxy'},
                                                 {'ndim': 4, 'shape': '1xy'},
                                                 {'ndim': 4, 'shape': '1xy'}]
                elif data_type == 'labels':
                    self.expected_shapes[key] = [{'ndim': 2, 'shape': 'xy'},
                                                 {'ndim': 3, 'shape': '1xy'}]
