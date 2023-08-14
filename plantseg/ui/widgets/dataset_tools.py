from enum import Enum
from pathlib import Path
from typing import Optional
from typing import Protocol

import napari
from magicgui import magicgui
from napari.layers import Labels, Image, Layer

from plantseg import PLANTSEG_MODELS_DIR
from plantseg.dataset_tools.dataset_handler import DatasetHandler, save_dataset, load_dataset
from plantseg.dataset_tools.dataset_handler import delete_dataset, change_dataset_location
from plantseg.dataset_tools.images import Image as PlantSegImage
from plantseg.dataset_tools.images import Labels as PlantSegLabels
from plantseg.dataset_tools.images import Stack, ImageSpecs, StackSpecs
from plantseg.ui.logging import napari_formatted_logging
from plantseg.utils import list_datasets, get_dataset_dict, delist_dataset

empty_dataset = ['none']
startup_list_datasets = list_datasets() or empty_dataset


class ImageType(Enum):
    IMAGE: str = 'image'
    LABELS: str = 'labels'


@magicgui(call_button='Initialize Dataset',
          dataset_name={'label': 'Dataset name',
                        'tooltip': f'Initialize an empty dataset with name model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing where the dataset will be created, '
                                  '{dataset_dir}/model_name/.'},
          dimensionality={'label': 'Dimensionality',
                          'choices': ['2D', '3D'],
                          'tooltip': f'Initialize an empty dataset with name model_name'},
          images_format={'label': 'Expected images format \n (Name, Channel, Type)',
                         'layout': 'vertical',
                         'tooltip': f'Define the expected images format for the dataset.\n'},
          is_sparse={'label': 'Is Dataset sparse?',
                     'tooltip': 'If checked, this info will be saved for training.'},
          )
def widget_create_dataset(dataset_name: str = 'my-dataset',
                          dataset_dir: Path = Path.home(),
                          dimensionality: str = '2D',
                          images_format: list[tuple[str, int, ImageType]] = (('raw', 1, ImageType.IMAGE),
                                                                             ('labels', 1, ImageType.LABELS)),
                          is_sparse: bool = False):
    if dataset_name in list_datasets():
        napari_formatted_logging(message='Dataset already exists.', thread='widget_create_dataset', level='warning')
        return None

    list_images = []
    for key, num_channels, im_format in images_format:

        if im_format == ImageType.IMAGE:
            image_spec = ImageSpecs(key=key,
                                    num_channels=num_channels,
                                    dimensionality=dimensionality,
                                    data_type='image')
            list_images.append(image_spec)
        elif im_format == ImageType.LABELS:
            assert num_channels == 1, 'Labels must have only one channel.'
            labels_spec = ImageSpecs(key=key,
                                     num_channels=1,
                                     dimensionality=dimensionality,
                                     data_type='labels',
                                     is_sparse=is_sparse)
            list_images.append(labels_spec)

        else:
            raise ValueError(f'Image format {im_format} not supported.')

    dataset_dir = dataset_dir / dataset_name

    stack_specs = StackSpecs(dimensionality=dimensionality,
                             list_specs=list_images)

    dataset = DatasetHandler(name=dataset_name,
                             dataset_dir=dataset_dir,
                             expected_stack_specs=stack_specs)

    save_dataset(dataset)
    return dataset


class Action(Protocol):
    name: str

    @staticmethod
    def edit(**kwargs):
        pass

    @staticmethod
    def on_action_changed():
        pass

    @staticmethod
    def on_edit_done():
        pass


class SingletonAction:
    _instance = None
    name: str = 'Abstract action'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    @staticmethod
    def edit(**kwargs):
        pass

    @staticmethod
    def on_action_changed():
        pass

    @staticmethod
    def on_edit_done():
        pass


class AddStack(SingletonAction):
    name = 'Add stack to dataset'

    @staticmethod
    def edit(dataset_name: str = startup_list_datasets[0],
             images: list[tuple[str, Optional[Layer]]] = (),
             new_stack_name: str = None,
             phase: str = 'train',
             is_sparse: bool = False,
             strict: bool = False,
             **kwargs):
        dataset = load_dataset(dataset_name)
        image_specs = dataset.expected_stack_specs.list_specs
        stack_specs = dataset.expected_stack_specs

        list_images = []
        for image_name, layer in images:
            reference_spec = [spec for spec in image_specs if spec.key == image_name][0]

            if isinstance(layer, Image) and reference_spec.data_type == 'image':
                image_data = layer.data
                image = PlantSegImage(image_data, spec=reference_spec)
            elif isinstance(layer, Labels) and reference_spec.data_type == 'labels':
                labels_data = layer.data
                reference_spec.is_sparse = is_sparse
                image = PlantSegLabels(labels_data, spec=reference_spec)
            else:
                napari_formatted_logging(message=f'Layer {image_name} is not of the expected type.',
                                         thread='AddStack',
                                         level='warning')
                return None

            list_images.append(image)

        stack_name = new_stack_name if new_stack_name != '' else images[0][1].name
        stack = Stack(*list_images, spec=stack_specs)
        dataset.add_stack(stack_name=stack_name, stack=stack, phase=phase)


class RemoveStack(SingletonAction):
    name = 'Remove stack from dataset'

    @staticmethod
    def edit(dataset_name: str, stack_name: str, **kwargs):
        dataset = load_dataset(dataset_name)
        if stack_name not in dataset.find_stacks_names():
            napari_formatted_logging(message=f'Stack {stack_name} does not exist.',
                                     thread='RemoveStack',
                                     level='warning')
            return None
        dataset.remove_stack(stack_name)


class RenameStack(SingletonAction):
    name = 'Rename stack from dataset'

    @staticmethod
    def edit(dataset_name: str, stack_name: str, new_stack_name: str, **kwargs):
        dataset = load_dataset(dataset_name)
        if stack_name not in dataset.find_stacks_names():
            napari_formatted_logging(message=f'Stack {stack_name} does not exist.',
                                     thread='RemoveStack',
                                     level='warning')
            return None
        dataset.rename_stack(stack_name, new_stack_name)


class ChangePhaseStack(SingletonAction):
    name = 'Change phase of stack from dataset'

    @staticmethod
    def edit(dataset_name: str, stack_name: str, new_phase: str, **kwargs):
        dataset = load_dataset(dataset_name)
        if stack_name not in dataset.find_stacks_names():
            napari_formatted_logging(message=f'Stack {stack_name} does not exist.',
                                     thread='RemoveStack',
                                     level='warning')
            return None
        dataset.change_phase_stack(stack_name, new_phase)


class LinkNewLocation(SingletonAction):
    name = 'Link dataset to a new folder'

    @staticmethod
    def edit(dataset_name: str, new_location: Path, **kwargs):
        if dataset_name not in list_datasets():
            napari_formatted_logging(message=f'Dataset {dataset_name} does not exist.',
                                     thread='LinkNewLocation',
                                     level='warning')
            return None

        if not new_location.exists():
            napari_formatted_logging(message=f'Location does not exist.',
                                     thread='LinkNewLocation',
                                     level='warning')
            return None

        change_dataset_location(dataset_name, new_location)


class DeleteDataset(SingletonAction):
    name = 'Delete dataset (cannot be undone!)'

    @staticmethod
    def edit(dataset_name: str, **kwargs):
        if dataset_name not in list_datasets():
            napari_formatted_logging(message=f'Dataset {dataset_name} does not exist.',
                                     thread='DeleteDataset',
                                     level='warning')
            return None
        dataset_dict = get_dataset_dict(dataset_name)
        assert dataset_dict is not None, f'Dataset {dataset_name} does not exist.'
        assert 'dataset_dir' in dataset_dict, f'Dataset {dataset_name} does not contain a dataset_dir.'
        assert Path(dataset_dict['dataset_dir']).exists(), f'Dataset {dataset_name} does not exist.'
        dataset_dir = dataset_dict['dataset_dir']
        delete_dataset(dataset_name, dataset_dir)


class DeListDataset(SingletonAction):
    name = 'De-List dataset (h5 will not be deleted)'

    @staticmethod
    def edit(dataset_name: str, **kwargs):
        if dataset_name not in list_datasets():
            napari_formatted_logging(message=f'Dataset {dataset_name} does not exist.',
                                     thread='DeListDataset',
                                     level='warning')
            return None
        delist_dataset(dataset_name)


class PrintDataset(SingletonAction):
    name = 'Print dataset information'

    @staticmethod
    def edit(dataset_name: str, **kwargs):
        if dataset_name not in list_datasets():
            napari_formatted_logging(message=f'Dataset {dataset_name} does not exist.',
                                     thread='PrintDataset',
                                     level='warning')
            return None
        dataset = load_dataset(dataset_name)
        napari_formatted_logging(message=f'Dataset infos:\n{dataset.info()}',
                                 thread='PrintDataset',
                                 level='info')


class VisualizeStack(SingletonAction):
    name = 'Visualize stack'

    @staticmethod
    def edit(viewer: napari.Viewer, dataset_name: str, stack_name: str, **kwargs):
        dataset = load_dataset(dataset_name)
        if stack_name not in dataset.find_stacks_names():
            napari_formatted_logging(message=f'Stack {stack_name} does not exist.',
                                     thread='RemoveStack',
                                     level='warning')
            return None

        stack, result, msg = dataset.get_stack_from_name(stack_name=stack_name)
        if result is False:
            napari_formatted_logging(message=msg,
                                     thread='OpenStack',
                                     level='warning')
            return None

        for spec in stack.list_specs:
            image = stack.data[spec.key]
            raw_data = image.load_data()
            if image.infos is not None:
                voxel_size, _, _, unit = image.infos

            else:
                voxel_size, unit = [1., 1., 1.], 'µm'

            metadata = {'original_voxel_size': voxel_size,
                        'voxel_size_unit': unit,
                        'root_name': stack_name}

            image_type = spec.data_type
            if image_type == 'image':
                viewer.add_image(raw_data, name=spec.key, scale=voxel_size, metadata=metadata)
            elif image_type == 'labels':
                viewer.add_labels(raw_data, name=spec.key, scale=voxel_size, metadata=metadata)


add_stack_action = AddStack()
visualize_stack_action = VisualizeStack()
remove_stack_action = RemoveStack()
rename_stack_action = RenameStack()
change_phase_stack_action = ChangePhaseStack()
print_dataset_action = PrintDataset()
link_new_location_action = LinkNewLocation()
delete_dataset_action = DeleteDataset()
delist_dataset_action = DeListDataset()

available_actions: dict[str, Action] = {action.name: action for action in [add_stack_action,
                                                                           visualize_stack_action,
                                                                           remove_stack_action,
                                                                           rename_stack_action,
                                                                           change_phase_stack_action,
                                                                           print_dataset_action,
                                                                           link_new_location_action,
                                                                           delete_dataset_action,
                                                                           delist_dataset_action]}


@magicgui(call_button=list(available_actions.values())[0].name,
          action={'label': 'Action',
                  'choices': list(available_actions.keys()),
                  'tooltip': f'Define if the stack will be added or removed from the dataset'},
          dataset_name={'label': 'Dataset name',
                        'choices': startup_list_datasets,
                        'tooltip': f'Choose the dataset to be edited'},
          phase={'label': 'Phase',
                 'choices': ['train', 'val', 'test'],
                 'tooltip': f'Define if the stack will be used for training, validation or testing'},
          is_sparse={'label': 'Is Stack sparse?',
                     'tooltip': 'If checked, this info will be saved for training.'},
          stack_name={'label': 'Stack name',
                      'tooltip': f'Name of the stack to be added to be edited',
                      'choices': ['', ]},
          new_stack_name={'label': 'New stack name',
                          'tooltip': f'Name of the stack to be added to be edited'},
          new_dataset_location={'label': 'New dataset location',
                                'mode': 'd',
                                'tooltip': f'New location of the dataset'},
          new_phase={'label': 'New phase',
                     'choices': ['train', 'val', 'test'],
                     'tooltip': f'Define if the stack will be used for training, validation or testing'},
          )
def widget_edit_dataset(viewer: napari.Viewer,
                        action: str = list(available_actions.keys())[0],
                        dataset_name: str = startup_list_datasets[0],
                        images: list[tuple[str, Optional[Layer]]] = (),
                        phase: str = 'train',
                        is_sparse: bool = False,
                        stack_name: str = '',
                        new_stack_name: str = '',
                        new_dataset_location: Path = None,
                        new_phase: str = 'val',
                        ) -> str:
    action_class = available_actions[action]

    kwargs = {'viewer': viewer,
              'dataset_name': dataset_name,
              'new_stack_name': new_stack_name,
              'new_dataset_location': new_dataset_location,
              'images': images,
              'phase': phase,
              'is_sparse': is_sparse,
              'stack_name': stack_name,
              'new_phase': new_phase,
              }
    action_class.edit(**kwargs)
    napari_formatted_logging(message=f'Action {action} applied to dataset {dataset_name}.',
                             thread='EditDataset',
                             level='info')
    return action


def safe_load_current_dataset(dataset_name: str = None) -> Optional[DatasetHandler]:
    _list_datasets = list_datasets()
    if len(_list_datasets) == 0:
        return None

    if dataset_name is None:
        dataset_name = widget_edit_dataset.dataset_name.value
        if dataset_name == empty_dataset[0]:
            return None

        if dataset_name not in _list_datasets:
            dataset_name = _list_datasets[0]

    return load_dataset(dataset_name)


def _update_dataset_name_choices(dataset: DatasetHandler = None):
    _list_datasets = list_datasets()

    if len(_list_datasets) == 0:
        widget_edit_dataset.dataset_name.choices = empty_dataset
        widget_edit_dataset.dataset_name.value = empty_dataset[0]
        return None

    if dataset is None:
        dataset = safe_load_current_dataset()

    widget_edit_dataset.dataset_name.choices = _list_datasets
    widget_edit_dataset.dataset_name.value = dataset.name


def _update_stack_name_choices(dataset: DatasetHandler = None):
    if dataset is None:
        dataset = safe_load_current_dataset()

    stacks_options = dataset.find_stacks_names()
    stacks_options = stacks_options if stacks_options else ['']
    widget_edit_dataset.stack_name.choices = stacks_options
    widget_edit_dataset.stack_name.value = stacks_options[0] if stacks_options else ''


def _update_images_choices(dataset: DatasetHandler = None):
    if dataset is None:
        dataset = safe_load_current_dataset()

    images_default = []
    for image in dataset.expected_stack_specs.list_specs:
        images_default.append((image.key, None))

    widget_edit_dataset.images.value = images_default


def _on_action_changed_add_stack():
    widget_edit_dataset.dataset_name.show()
    widget_edit_dataset.phase.show()
    widget_edit_dataset.is_sparse.show()
    widget_edit_dataset.images.show()
    widget_edit_dataset.new_stack_name.show()

    widget_edit_dataset.new_stack_name.value = ''

    current_dataset = safe_load_current_dataset()
    if current_dataset is None:
        return None
    _update_dataset_name_choices(current_dataset)
    _update_images_choices(current_dataset)


add_stack_action.on_action_changed = _on_action_changed_add_stack


def _on_action_changed_remove_stack():
    widget_edit_dataset.stack_name.show()
    widget_edit_dataset.dataset_name.show()

    current_dataset = safe_load_current_dataset()
    if current_dataset is None:
        return None
    _update_dataset_name_choices(current_dataset)
    _update_stack_name_choices(current_dataset)


def _on_action_rename_stack():
    widget_edit_dataset.stack_name.show()
    widget_edit_dataset.dataset_name.show()
    widget_edit_dataset.new_stack_name.show()

    current_dataset = safe_load_current_dataset()
    if current_dataset is None:
        return None
    _update_dataset_name_choices(current_dataset)
    _update_stack_name_choices(current_dataset)


def _on_action_change_phase_stack():
    widget_edit_dataset.stack_name.show()
    widget_edit_dataset.dataset_name.show()
    widget_edit_dataset.new_phase.show()

    current_dataset = safe_load_current_dataset()
    if current_dataset is None:
        return None
    _update_dataset_name_choices(current_dataset)
    _update_stack_name_choices(current_dataset)


visualize_stack_action.on_action_changed = _on_action_changed_remove_stack
visualize_stack_action.on_edit_done = _update_stack_name_choices

rename_stack_action.on_action_changed = _on_action_rename_stack
rename_stack_action.on_edit_done = _update_stack_name_choices

remove_stack_action.on_action_changed = _on_action_changed_remove_stack
remove_stack_action.on_edit_done = _update_stack_name_choices

change_phase_stack_action.on_action_changed = _on_action_change_phase_stack
change_phase_stack_action.on_edit_done = _update_stack_name_choices


def _on_action_changed_delete_dataset():
    widget_edit_dataset.dataset_name.show()

    current_dataset = safe_load_current_dataset()
    if current_dataset is None:
        return None
    _update_dataset_name_choices(current_dataset)


delete_dataset_action.on_action_changed = _on_action_changed_delete_dataset
delete_dataset_action.on_edit_done = _update_dataset_name_choices

delist_dataset_action.on_action_changed = _on_action_changed_delete_dataset
delist_dataset_action.on_edit_done = _update_dataset_name_choices

print_dataset_action.on_action_changed = _on_action_changed_delete_dataset
print_dataset_action.on_edit_done = _update_dataset_name_choices


def _hide_all():
    widget_edit_dataset.dataset_name.hide()
    widget_edit_dataset.stack_name.hide()
    widget_edit_dataset.new_stack_name.hide()
    widget_edit_dataset.new_dataset_location.hide()
    widget_edit_dataset.dataset_name.hide()
    widget_edit_dataset.phase.hide()
    widget_edit_dataset.is_sparse.hide()
    widget_edit_dataset.images.hide()
    widget_edit_dataset.new_phase.hide()


@widget_create_dataset.called.connect
def _create_dataset_is_called(dataset: DatasetHandler):
    _update_dataset_name_choices(dataset)
    _update_stack_name_choices(dataset)

    widget_create_dataset.dataset_name.value = ''


@widget_edit_dataset.dataset_name.changed.connect
def update_dataset_name(dataset_name: str):
    dataset = safe_load_current_dataset(dataset_name)
    if dataset is None:
        return None

    widget_edit_dataset.is_sparse.value = dataset.is_sparse
    _update_stack_name_choices(dataset)
    _update_images_choices(dataset)


@widget_edit_dataset.action.changed.connect
def update_action(action: str):
    _hide_all()

    if action in available_actions.keys():
        action_class = available_actions[action]
        action_class.on_action_changed()
        widget_edit_dataset.call_button.text = action_class.name


@widget_edit_dataset.called.connect
def _on_done(action: str):
    if action is None:
        return None

    if action in available_actions.keys():
        action_class = available_actions[action]
        action_class.on_edit_done()


_hide_all()
list(available_actions.values())[0].on_action_changed()
