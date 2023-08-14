from pathlib import Path
from shutil import rmtree
from typing import Union, Protocol
from warnings import warn

from plantseg.dataset_tools.images import Stack, StackSpecs
from plantseg.io.h5 import H5_EXTENSIONS
from plantseg.utils import dump_dataset_dict, get_dataset_dict, delist_dataset, list_datasets


class DatasetValidator(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, dataset: object) -> tuple[bool, str]:
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


class DatasetHandler:
    """
    DatasetHandler is a class that contains all the information about a dataset.
    It is used to create a dataset from a directory, to save a dataset to a directory and manage the dataset.
    """
    default_phases: tuple[str, ...] = ('train', 'val', 'test')
    train: list[str]
    val: list[str]
    test: list[str]
    dimensionality: str
    default_file_formats = H5_EXTENSIONS

    def __init__(self,
                 name: str,
                 dataset_dir: Union[str, Path],
                 expected_stack_specs: StackSpecs):

        assert isinstance(name, str), 'name must be a string'
        self.name = name

        assert isinstance(dataset_dir, (str, Path)), 'dataset_dir must be a string or a Path'
        self.dataset_dir = Path(dataset_dir)

        self.train = []
        self.val = []
        self.test = []

        self.expected_stack_specs = expected_stack_specs
        self.init_datastructure()
        self.update_stack_from_disk()
        self.is_sparse = self._is_sparse()

    def init_datastructure(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for phase in self.default_phases:
            (self.dataset_dir / phase).mkdir(exist_ok=True)

    @property
    def dimensionality(self) -> str:
        return self.expected_stack_specs.dimensionality

    @classmethod
    def from_dict(cls, dataset_dict: dict):
        """
        Create a DatasetHandler from a dictionary.
        """
        assert 'name' in dataset_dict.keys(), 'Dataset name not found'
        assert 'dataset_dir' in dataset_dict.keys(), 'Dataset directory not found'
        assert 'stack_specs' in dataset_dict.keys(), 'Dataset stack_specs not found'
        assert 'list_specs' in dataset_dict['stack_specs'].keys(), 'Dataset list_specs not found'

        name = dataset_dict['name']
        dataset_dir = dataset_dict['dataset_dir']
        stack_specs = StackSpecs.from_dict(dataset_dict['stack_specs'])
        return cls(name=name, dataset_dir=dataset_dir, expected_stack_specs=stack_specs)

    def to_dict(self) -> dict:
        """
        Convert a DatasetHandler to a dictionary for serialization.
        """
        dataset_dict = {
            'name': self.name,
            'dataset_dir': str(self.dataset_dir),
            'stack_specs': self.expected_stack_specs.to_dict()
        }
        return dataset_dict

    def _is_sparse(self, stacks: list[Stack] = None) -> bool:
        if stacks is None:
            stacks = self.get_stacks()
        return all([stack.is_sparse for stack in stacks])

    def __repr__(self) -> str:
        return f'DatasetHandler {self.name}, location: {self.dataset_dir}'

    def info(self) -> str:
        """
        Nice print of the dataset information.
        """
        info = f'{self.__repr__()}:\n'
        info += f'Dimensionality: {self.expected_stack_specs.dimensionality}\n'
        info += f'Is sparse: {self.is_sparse}\n'
        info += f'Num of stacks: {len(self.find_stacks_names())} (train: {len(self.train)}, val: {len(self.val)}, ' \
                f'test: {len(self.test)}) \n'
        return info

    def validate(self, *dataset_validators: DatasetValidator) -> tuple[bool, str]:
        """
        Validate the dataset using the dataset validators.
        Returns:
            (bool, str): a boolean (True for success) and a message with the result of the validation
        """
        return ComposeDatasetValidators(*dataset_validators)(self)

    def get_stack(self, path: Union[str, Path]) -> tuple[Stack, bool, str]:
        """
        Get a stack from the dataset.
        Args:
            path: path to the stack
        Returns:
            Stack: the stack
        """
        return Stack.from_h5(path=path, expected_stack_specs=self.expected_stack_specs)

    def get_stack_from_name(self, stack_name: str) -> tuple[Stack, bool, str]:
        for h5 in self.find_stored_files():
            if stack_name == f'{h5.parent.name}/{h5.stem}':
                return self.get_stack(h5)

        raise ValueError(f'Stack {stack_name} not found in dataset {self.name}')

    def update_stack_from_disk(self, phase: str = None):
        """
        Update the stacks in the dataset from the disk.
        """
        if phase is None:
            phases = self.default_phases
        else:
            phases = [phase]

        is_sparse_phase = []
        for phase in phases:
            stacks_found = self.get_stacks(phase=phase)
            setattr(self, phase, stacks_found)
            is_sparse_phase.append(self._is_sparse(stacks_found))

        self.is_sparse = all(is_sparse_phase)

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
            phase_found_files = []
            phase_dir = self.dataset_dir / phase
            assert phase_dir.exists(), f'Phase {phase} not found in {self.dataset_dir}'

            for file_format in file_formats:
                stacks_found = [file for file in phase_dir.glob(f'*{file_format}')]
                phase_found_files.extend(stacks_found)

            phase_found_files = sorted(phase_found_files, key=lambda x: x.stem)
            found_files.extend(phase_found_files)
        return found_files

    def find_stacks_names(self, phase: str = None) -> list[str]:
        """
        Find the name of the stacks in the dataset directory.
        """
        stacks = self.find_stored_files(phase=phase)
        return [f'{stack.parent.name}/{stack.stem}' for stack in stacks]

    def get_stacks(self, phase: str = None) -> list[Stack]:
        """
        Get the stacks in the dataset directory.
        """
        stacks = self.find_stored_files(phase=phase)
        all_stacks = []
        for stack in stacks:
            stack, result, msg = self.get_stack(stack)
            if result:
                all_stacks.append(stack)
            else:
                warn(f'Stack {stack} seems to not be compatible with the dataset specs. Error {msg}, skipping it.')

        return all_stacks

    def add_stack(self, stack_name: str,
                  phase: str,
                  stack: Stack,
                  unique_name=True):
        """
        Add a stack to the dataset.
        Args:
            stack_name: string with the name of the stack
            phase: string with the phase of the dataset (train, val, test)
            stack: dictionary with the data to be saved in the stack
                {'raw': raw_data, 'labels': labels_data, etc...}
            unique_name: if True, the stack name will be changed to a unique name if already exists,
                otherwise it will error out.

        Returns: None
        """
        phase_dir = self.dataset_dir / phase
        stack_path = phase_dir / f'{stack_name}.h5'
        idx = 1

        while stack_path.exists() and unique_name:
            if stack_name.find('_') == -1:
                stack_name += f'_{idx}'
            else:
                *name_base, idx_name = stack_name.split('_')
                name_base = '_'.join(name_base)
                stack_name = f'{name_base}_{idx}'

            stack_path = phase_dir / f'{stack_name}.h5'
            idx += 1

        result, msg = stack.check_compatibility(self.expected_stack_specs)
        if result:
            stack.dump_to_h5(stack_path)
            return None

        raise ValueError(f'Could not add stack to dataset. {msg}')

    def remove_stack(self, stack_name: str):
        """
        Remove a stack from the dataset.
        Args:
            stack_name: string with the name of the stack

        Returns: None
        """
        for phase in self.default_phases:
            stacks = self.find_stacks_names(phase=phase)
            if stack_name in stacks:
                stack_path = self.dataset_dir / f'{stack_name}.h5'
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
            stacks = self.find_stacks_names(phase=phase)
            if stack_name in stacks:
                stack_path = self.dataset_dir / f'{stack_name}.h5'
                if stack_path.exists():
                    new_stack_path = self.dataset_dir / phase / f'{new_name}.h5'
                    stack_path.rename(new_stack_path)
                    self.update_stack_from_disk(phase=phase)
                    return None
                else:
                    raise FileNotFoundError(f'Stack {stack_name} not found in {phase} phase.')

        raise ValueError(f'Stack {stack_name} not found in dataset {self.name}.')

    def change_phase_stack(self, stack_name: str, new_phase: str):
        """
        Change the phase of a stack in the dataset.
        Args:
            stack_name: string with the name of the stack
            new_phase: string with the new phase of the stack

        Returns: None
        """
        assert new_phase in self.default_phases, f'Phase {new_phase} not found in dataset {self.name}.'
        for phase in self.default_phases:
            stacks = self.find_stacks_names(phase=phase)
            if stack_name in stacks:
                stack_path = self.dataset_dir / f'{stack_name}.h5'
                if stack_path.exists():
                    if phase == new_phase:
                        return None

                    stack_name = stack_name.split('/')[-1]
                    new_stack_path = self.dataset_dir / new_phase / f'{stack_name}.h5'
                    stack_path.rename(new_stack_path)
                    self.update_stack_from_disk()
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


def delete_dataset(dataset_name: str, dataset_dir: Union[str, Path]):
    """
    Delete a dataset from the user dataset config file and delete all the files
    Args:
        dataset_name: string with the name of the dataset
        dataset_dir: path to the dataset directory

    Returns: None
    """
    delist_dataset(dataset_name)
    rmtree(dataset_dir)


def change_dataset_location(dataset_name: str, new_location: Union[str, Path]):
    """
    Change the location of a dataset in the user dataset config file and move all the files
    Args:
        dataset_name: string with the name of the dataset
        new_location: new location of the dataset

    Returns:
        None
    """
    new_location = Path(new_location)
    if not new_location.exists():
        raise ValueError(f'New location {new_location} does not exist.')

    assert new_location.is_dir(), f'New location {new_location} is not a directory.'
    dataset_dict = get_dataset_dict(dataset_name)
    dataset_dict['dataset_dir'] = str(new_location)
    dataset = DatasetHandler.from_dict(dataset_dict)
    save_dataset(dataset)
