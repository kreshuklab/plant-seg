from enum import Enum
from pathlib import Path
from typing import Optional

from magicgui import magicgui
from napari.layers import Labels, Image, Layer

from plantseg import PLANTSEG_MODELS_DIR
from plantseg.dataset_tools.dataset_handler import DatasetHandler, save_dataset, load_dataset
from plantseg.dataset_tools.images import Image as PlantSegImage
from plantseg.dataset_tools.images import Labels as PlantSegLabels
from plantseg.dataset_tools.images import Stack, ImageSpecs, StackSpecs
from plantseg.ui.logging import napari_formatted_logging
from plantseg.utils import list_datasets

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
                         'tooltip': f'Initialize an empty dataset with name model_name'},
          is_sparse={'label': 'Sparse dataset',
                     'tooltip': 'If checked, the dataset will be saved in sparse format.'},
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


def _add_stack(dataset_name: str = startup_list_datasets[0],
               images: list[tuple[str, Optional[Layer]]] = (),
               phase: str = 'train',
               is_sparse: bool = False,
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
            raise ValueError(f'Layer type {type(layer)} not supported.')

        list_images.append(image)

    stack_name = images[0][1].name
    stack = Stack(*list_images, spec=stack_specs)
    dataset.add_stack(stack_name=stack_name, stack=stack, phase=phase)


def _remove_stack(dataset_name, stack_name: str, **kwargs):
    dataset = load_dataset(dataset_name)
    dataset.remove_stack(stack_name)


available_modes = {
    'Add stack to dataset': _add_stack,
    'Remove stack from dataset': _remove_stack,
    'Remove dataset': None,
    'De-list dataset': None,
    'Move dataset': None,
    'Rename dataset': None
}


@magicgui(call_button='Edit Dataset',
          action={'label': 'Action',
                  'choices': list(available_modes.keys()),
                  'tooltip': f'Define if the stack will be added or removed from the dataset'},
          dataset_name={'label': 'Dataset name',
                        'choices': startup_list_datasets,
                        'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          phase={'label': 'Phase',
                 'choices': ['train', 'val', 'test'],
                 'tooltip': f'Define if the stack will be used for training, validation or testing'},
          is_sparse={'label': 'Sparse dataset',
                     'tooltip': 'If checked, the dataset will be saved in sparse format.'},
          stack_name={'label': 'Stack name',
                      'tooltip': f'Name of the stack to be added to be edited',
                      'choices': ['', ]},
          )
def widget_edit_dataset(action: str = list(available_modes.keys())[0],
                        dataset_name: str = startup_list_datasets[0],
                        images: list[tuple[str, Optional[Layer]]] = (),
                        phase: str = 'train',
                        is_sparse: bool = False,
                        stack_name: str = '') -> str:
    func = available_modes[action]
    kwargs = {'dataset_name': dataset_name,
              'images': images,
              'phase': phase,
              'is_sparse': is_sparse,
              'stack_name': stack_name}
    func(**kwargs)
    return action


@widget_create_dataset.called.connect
def update_dataset_name(dataset: DatasetHandler):
    if dataset is None:
        return None

    widget_edit_dataset.dataset_name.choices = list_datasets()
    widget_edit_dataset.dataset_name.value = dataset.name


def _update_stack_name_choices(dataset: DatasetHandler = None):
    if dataset is None:
        dataset = load_dataset(widget_edit_dataset.dataset_name.value)
    stacks_options = dataset.find_stacks_names()
    stacks_options = stacks_options if stacks_options else ['']
    widget_edit_dataset.stack_name.choices = stacks_options
    widget_edit_dataset.stack_name.value = stacks_options[0] if stacks_options else ''


def _update_images_choices(dataset: DatasetHandler = None):
    if len(list_datasets()) == 0:
        return None

    if dataset is None:
        dataset = load_dataset(widget_edit_dataset.dataset_name.value)

    images_default = []
    for image in dataset.expected_stack_specs.list_specs:
        images_default.append((image.key, None))

    widget_edit_dataset.images.value = images_default


_update_images_choices()


@widget_edit_dataset.dataset_name.changed.connect
def update_dataset_name(dataset_name: str):
    dataset = load_dataset(dataset_name)
    widget_edit_dataset.phase.choices = dataset.default_phases
    widget_edit_dataset.is_sparse.value = dataset.is_sparse

    _update_images_choices(dataset)
    _update_stack_name_choices(dataset)


def _add_stack_callback():
    widget_edit_dataset.images.show()
    widget_edit_dataset.phase.show()
    widget_edit_dataset.stack_name.hide()


def _remove_stack_callback():
    widget_edit_dataset.images.hide()
    widget_edit_dataset.phase.hide()
    widget_edit_dataset.is_sparse.hide()
    widget_edit_dataset.stack_name.show()

    _update_stack_name_choices()


_add_stack_callback()


available_modes_callbacks = {
    'Add stack to dataset': _add_stack_callback,
    'Remove stack from dataset': _remove_stack_callback,
}


@widget_edit_dataset.action.changed.connect
def update_mode(action: str):
    if action in available_modes_callbacks.keys():
        available_modes_callbacks[action]()


def _remove_stack_update_choices():
    _update_stack_name_choices()


available_actions_on_done = {
    'Remove stack from dataset': _remove_stack_update_choices
}


@widget_edit_dataset.called.connect
def update_state_after_edit(action: str):
    if action in available_actions_on_done.keys():
        available_actions_on_done[action]()
