from pathlib import Path
from shutil import rmtree
from typing import Union, Protocol
from warnings import warn

from plantseg.dataset_tools.images import Stack
from plantseg.io.h5 import H5_EXTENSIONS
from plantseg.utils import dump_dataset_dict, get_dataset_dict, delist_dataset, list_datasets


class DatasetValidator(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, dataset: object) -> tuple[bool, str]:
        ...


class StackValidator(Protocol):
    def __init__(self, dataset: object):
        ...

    def __call__(self, stack_path: Union[str, Path]) -> tuple[bool, str]:
        ...


class ComposeDatasetValidators:
    """
    Compose multiple dataset validators into a single one.
    """
    success_msg = 'All tests passed.'

    def __init__(self, *validators: DatasetValidator):
        self.validators = validators

    def __call__(self, dataset: object) -> tuple[bool, str]:
        return self.apply(dataset)

    def apply(self, dataset: object) -> tuple[bool, str]:
        """
        Apply all the validators to the dataset.
        Args:
            dataset: dataset to validate
        Returns:
            tuple[bool, str]: (valid, msg) where valid is True if all the tests passed, False otherwise.
        """
        for validator in self.validators:
            valid, msg = validator(dataset)
            if not valid:
                return valid, msg
        return True, self.success_msg

    def batch_apply(self, list_dataset: list[object]) -> tuple[bool, str]:
        """
        Apply all the validators to a list of datasets.
        Args:
            list_dataset: list of datasets to validate
        Returns:
            tuple[bool, str]: (valid, msg) where valid is True if all the tests passed, False otherwise.
        """
        for dataset in list_dataset:
            valid, msg = self.apply(dataset)
            if not valid:
                msg = f'Validation failed for {dataset}.\nWith msg: {msg}'
                return False, msg

        return True, self.success_msg


class ComposeStackValidators:
    """
    Compose multiple stack validators into a single one.
    """
    success_msg = 'All tests passed.'

    def __init__(self, *validators: StackValidator):
        self.validators = validators

    def __call__(self, stack_path: Union[str, Path]) -> tuple[bool, str]:
        return self.apply(stack_path)

    def apply(self, stack_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Apply all the validators to the stack.
        Args:
            stack_path: path to the stack to validate

        Returns:
            tuple[bool, str]: (valid, msg) where valid is True if all the tests passed, False otherwise.

        """
        for validator in self.validators:
            valid, msg = validator(stack_path)
            if not valid:
                return valid, msg
        return True, self.success_msg

    def batch_apply(self, list_stack_path: list[Union[str, Path]]) -> tuple[bool, str]:
        """
        Apply all the validators to a list of stacks.
        Args:
            list_stack_path: list of paths to the stacks to validate

        Returns:
            tuple[bool, str]: (valid, msg) where valid is True if all the tests passed, False otherwise.

        """
        for stack_path in list_stack_path:
            valid, msg = self.apply(stack_path)
            if not valid:
                msg = f'Validation failed for {stack_path}.\nWith msg: {msg}'
                return False, msg

        return True, self.success_msg


class DatasetHandler:
    """
    DatasetHandler is a class that contains all the information about a dataset.
    It is used to create a dataset from a directory, to save a dataset to a directory and manage the dataset.
    """
    name: str
    keys: tuple[str, ...]
    default_phases: tuple[str, ...]
    default_file_formats: tuple[str, ...]
    default_phases: tuple[str, ...] = ('train', 'val', 'test')
    train: list[str]
    val: list[str]
    test: list[str]

    _default_keys = {'task': None,
                     'dimensionality': None,  # 2D or 3D
                     'image_channels': None,
                     'keys': ('raw', 'labels'),  # keys of the h5 file (raw, labels, etc. )
                     'is_sparse': False,
                     'default_file_formats': H5_EXTENSIONS
                     }

    def __init__(self, name: str, dataset_dir: Union[str, Path], **kwargs):
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train = []
        self.val = []
        self.test = []

        for atr, default in self._default_keys.items():
            setattr(self, atr, default)

        for atr, value in kwargs.items():
            if atr in self._default_keys.keys():
                setattr(self, atr, value)
            else:
                raise ValueError(f'Attribute {atr} does not exists for {self.__class__.__name__}.')

    @classmethod
    def from_dict(cls, dataset_dict: dict):
        """
        Create a DatasetHandler from a dictionary.
        """
        assert 'name' in dataset_dict.keys(), 'Dataset name not found'
        assert 'dataset_dir' in dataset_dict.keys(), 'Dataset directory not found'
        dataset = cls(name=dataset_dict['name'], dataset_dir=dataset_dict['dataset_dir'])

        for atr, default in cls._default_keys.items():
            if atr in dataset_dict.keys():
                setattr(dataset, atr, dataset_dict.get(atr))
            else:
                warn(f'Attribute {atr} not found in dataset {dataset.name}. Setting to default value {default}')
                setattr(dataset, atr, default)

        dataset.update_stack_from_disk()
        return dataset

    def to_dict(self) -> dict:
        """
        Convert a DatasetHandler to a dictionary for serialization.
        """
        dataset_dict = {'name': self.name, 'dataset_dir': str(self.dataset_dir)}
        for atr in self._default_keys.keys():
            dataset_dict[atr] = getattr(self, atr)
        return dataset_dict

    def __repr__(self) -> str:
        return f'DatasetHandler {self.name}, location: {self.dataset_dir}'

    def info(self) -> str:
        """
        Nice print of the dataset information.
        """
        info = f'{self.__repr__()}:\n'
        for atr in self._default_keys.keys():
            if atr in self.default_phases:
                info += f'    {atr}: #{len(getattr(self, atr))} stacks\n'
            else:
                info += f'    {atr}: {getattr(self, atr)}\n'
        return info

    def validate(self, *dataset_validators: DatasetValidator) -> tuple[bool, str]:
        """
        Validate the dataset using the dataset validators.
        Returns:
            (bool, str): a boolean (True for success) and a message with the result of the validation
        """
        return ComposeDatasetValidators(*dataset_validators)(self)

    def validate_stack(self, *stack_validators: StackValidator) -> tuple[bool, str]:
        """
        Validate all the stacks in the dataset using the stack validators.
        Returns:
            (bool, str): a boolean (True for success) and a message with the result of the validation
        """
        files = self.find_stored_files()
        return ComposeStackValidators(*stack_validators).batch_apply(files)

    def update_stack_from_disk(self, *validators: StackValidator, phase: str = None):
        """
        Update the stacks in the dataset from the disk.
        """
        if phase is None:
            phases = self.default_phases
        else:
            phases = [phase]

        for phase in phases:
            stacks = self.find_stored_files(phase=phase)
            result, msg = ComposeStackValidators(*validators).batch_apply(stacks)
            if result:
                stacks = [stack.name for stack in stacks]
                setattr(self, phase, stacks)
            else:
                warn(f'Update failed for {phase} phase. {msg}')

    def find_stored_files(self, phase: str = None, ignore_default_file_format: bool = False) -> list[Path]:
        """
        Find files in the dataset directory, by default it will only look at the defaults file extensions.
        Args:
            phase: a string with the phase of the dataset, if None all phases are searched
            ignore_default_file_format: set to True to ignore the default file format
        Returns:
            a list of paths to the stacks found
        """
        if phase is None:
            phases = self.default_phases
        elif isinstance(phase, str):
            assert phase in self.default_phases, f'Phase {phase} not found in {self.default_phases}'
            phases = (phase,)

        else:
            raise ValueError(f'Phase must be a string or None, found {type(phase)}')

        found_files = []
        file_formats = self.default_file_formats if not ignore_default_file_format else ('*',)

        for phase in phases:
            phase_dir = self.dataset_dir / phase
            assert phase_dir.exists(), f'Phase {phase} not found in {self.dataset_dir}'

            for file_format in file_formats:
                stacks_found = [file for file in phase_dir.glob(f'*{file_format}')]
                found_files.extend(stacks_found)

        return found_files

    def find_stacks(self, phase: str = None) -> list[str]:
        """
        Find the name of the stacks in the dataset directory.
        """
        stacks = self.find_stored_files(phase=phase)
        return [stack.stem for stack in stacks]

    def add_stack(self, stack_name: str,
                  phase: str,
                  data: Stack,
                  unique_name=True):
        """
        Add a stack to the dataset.
        Args:
            stack_name: string with the name of the stack
            phase: string with the phase of the dataset (train, val, test)
            data: dictionary with the data to be saved in the stack
                {'raw': raw_data, 'labels': labels_data, etc...}
            unique_name: if True, the stack name will be changed to a unique name if already exists,
                otherwise it will error out.

        Returns: None
        """
        phase_dir = self.dataset_dir / phase
        stack_path = phase_dir / f'{stack_name}.h5'
        idx = 1
        while stack_path.exists() and unique_name:
            stack_name += f'_{idx}'
            stack_path = phase_dir / f'{stack_name}.h5'
            idx += 1

        data.dump_to_h5(stack_path)

    def remove_stack(self, stack_name: str):
        """
        Remove a stack from the dataset.
        Args:
            stack_name: string with the name of the stack

        Returns: None
        """
        for phase in self.default_phases:
            stacks = self.find_stacks(phase=phase)
            if stack_name in stacks:
                stack_path = self.dataset_dir / phase / f'{stack_name}.h5'
                if stack_path.exists():
                    stack_path.unlink()
                    self.update_stack_from_disk(phase=phase)
                    return None
                else:
                    raise FileNotFoundError(f'Stack {stack_name} not found in {phase} phase.')

        raise ValueError(f'Stack {stack_name} not found in dataset {self.name}.')

    def rename_stack(self, stack_name: str, new_name: str):
        """
        Rename a stack from the dataset.
        Args:
            stack_name: string with the name of the stack
            new_name: string with the new name of the stack

        Returns: None
        """
        for phase in self.default_phases:
            stacks = self.find_stacks(phase=phase)
            if stack_name in stacks:
                stack_path = self.dataset_dir / phase / f'{stack_name}.h5'
                if stack_path.exists():
                    new_stack_path = self.dataset_dir / phase / f'{new_name}.h5'
                    stack_path.rename(new_stack_path)
                    self.update_stack_from_disk(phase=phase)
                    return None
                else:
                    raise FileNotFoundError(f'Stack {stack_name} not found in {phase} phase.')

        raise ValueError(f'Stack {stack_name} not found in dataset {self.name}.')


def load_dataset(dataset_name: str) -> DatasetHandler:
    """
    Load a dataset from the user dataset config file.
    Args:
        dataset_name: string with the name of the dataset

    Returns:
        a DatasetHandler object
    """
    if dataset_name not in list_datasets():
        raise ValueError(f'Dataset {dataset_name} not found in existing datasets: {list_datasets()}')

    dataset_dict = get_dataset_dict(dataset_name)
    dataset = DatasetHandler.from_dict(dataset_dict)
    return dataset


def save_dataset(dataset: DatasetHandler):
    """
    Save a dataset to the user dataset config file.
    Args:
        dataset: a DatasetHandler object

    Returns: None
    """
    dump_dataset_dict(dataset.name, dataset.to_dict())


def delete_dataset(dataset: DatasetHandler):
    """
    Delete a dataset from the user dataset config file and delete all the files
    Args:
        dataset: a DatasetHandler object

    Returns: None
    """
    delist_dataset(dataset.name)
    rmtree(dataset.dataset_dir)
