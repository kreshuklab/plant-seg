from pathlib import Path

from magicgui import magicgui
from napari.layers import Labels, Image

from plantseg import PLANTSEG_MODELS_DIR
from plantseg.io import create_h5
from plantseg.utils import list_datasets, save_dataset, get_dataset, delete_dataset
from plantseg.viewer.logging import napari_formatted_logging

empty_dataset = ['none']
startup_list_datasets = list_datasets() or empty_dataset


@magicgui(call_button='Initialize Dataset',
          dataset_name={'label': 'Dataset name',
                        'tooltip': f'Initialize an empty dataset with name model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing where the dataset will be created, '
                                  '{dataset_dir}/model_name/.'}
          )
def widget_create_dataset(dataset_name: str = 'my-dataset', dataset_dir: Path = Path.home()):
    dataset_dir = dataset_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    new_dataset = {'name': dataset_name,
                   'dataset_dir': str(dataset_dir),
                   'task': None,
                   'dimensionality': None,  # 2D or 3D
                   'image_channels': None,
                   'image_key': 'raw',
                   'labels_key': 'labels',
                   'is_sparse': False,
                   'train': [],
                   'val': [],
                   'test': [],
                   }

    if dataset_name not in list_datasets():
        save_dataset(dataset_name, new_dataset)
        return new_dataset

    raise ValueError(f'Dataset {dataset_name} already exists.')


@magicgui(call_button='Create Dataset',
          dataset_name={'label': 'Dataset name',
                        'choices': startup_list_datasets,
                        'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          phase={'label': 'Phase',
                 'choices': ['train', 'val', 'test'],
                 'tooltip': f'Define if the stack will be used for training, validation or testing'},
          is_sparse={'label': 'Sparse dataset',
                     'tooltip': 'If checked, the dataset will be saved in sparse format.'}
          )
def widget_add_stack(dataset_name: str = startup_list_datasets[0],
                     image: Image = None,
                     labels: Labels = None,
                     phase: str = 'train',
                     is_sparse: bool = False):
    dataset_config = get_dataset(dataset_name)

    if image is None or labels is None:
        napari_formatted_logging(message=f'To add a stack to the dataset, please select an image and a labels layer.',
                                 thread='widget_add_stack',
                                 level='warning')
        return None

    if is_sparse:
        # if a single dataset is sparse, all the others should be threaded as sparse
        dataset_config['is_sparse'] = True

    image_data = image.data
    labels_data = labels.data

    # Validation of the image and labels data
    # check if the image and labels have the same shape,
    # dimensionality and number of channels as the rest of the dataset

    if image_data.ndim == 3:
        image_channels = 1
        dimensionality = '2D' if image_data.shape[0] == 1 else '3D'
        assert image_data.shape == labels_data.shape, f'Image and labels should have the same shape, found ' \
                                                      f'{image_data.shape} and {labels_data.shape}.'

    elif image_data.ndim == 4:
        image_channels = image_data.shape[0]
        dimensionality = '2D' if image_data.shape[1] == 1 else '3D'
        assert image_data.shape[1:] == labels_data.shape, f'Image and labels should have the same shape, found ' \
                                                          f'{image_data.shape} and {labels_data.shape}.'

    else:
        raise ValueError(f'Image data should be 3D or multichannel 3D, found {image_data.ndim}D.')

    dataset_image_channels = dataset_config['image_channels']
    if dataset_image_channels is None:
        dataset_config['image_channels'] = image_channels
    elif dataset_image_channels != image_channels:
        raise ValueError(f'Image data should have {dataset_image_channels} channels, found {image_channels}.')

    dataset_dimensionality = dataset_config['dimensionality']
    if dataset_dimensionality is None:
        dataset_config['dimensionality'] = dimensionality
    elif dataset_dimensionality != dimensionality:
        raise ValueError(f'Image data should be {dataset_dimensionality}, found {dimensionality}.')

    if is_sparse:
        dataset_config['is_sparse'] = True

    # Check if the stack name already exists in the dataset
    # If so, add a number to the end of the name until it is unique
    stack_name = image.name
    existing_stacks = dataset_config[phase]

    idx = 0
    while True:
        if stack_name in existing_stacks:
            stack_name = f'{stack_name}_{idx}'
        else:
            break
        idx += 1

    dataset_config[phase].append(stack_name)

    # Save the data to disk
    dataset_dir = Path(dataset_config['dataset_dir']) / phase
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_path = str(dataset_dir / f'{stack_name}.h5')
    create_h5(image_path, image_data, key=dataset_config['image_key'])
    create_h5(image_path, labels_data, key=dataset_config['labels_key'])
    save_dataset(dataset_name, dataset_config)
    napari_formatted_logging(message=f'Stack {stack_name} added to dataset {dataset_name}.',
                             thread='widget_add_stack',
                             level='info')


@magicgui(call_button='Validata Dataset',
          dataset_name={'label': 'Dataset name',
                        'choices': startup_list_datasets,
                        'tooltip': f'Name of the dataset to be validated'},
          )
def widget_validata_dataset(dataset_name: str = startup_list_datasets[0]):
    dataset_config = get_dataset(dataset_name)

    # check all stacks are present
    dataset_dir = Path(dataset_config['dataset_dir'])
    for phase in ['train', 'val', 'test']:
        phase_dir = dataset_dir / phase
        stacks_expected = dataset_config[phase]
        stacks_found = [file.stem for file in phase_dir.glob('*.h5')]
        if len(stacks_found) != len(stacks_expected):
            napari_formatted_logging(message=f'Found {len(stacks_found)} stacks in {phase} phase, '
                                             f'expected {len(stacks_expected)}.',
                                     thread='widget_validata_dataset',
                                     level='warning')

            dataset_config[phase] = stacks_found

    # check all stacks have the same shape and dimensionality
    for key, value in dataset_config.items():
        napari_formatted_logging(message=f'Dataset info {key}: {value}',
                                 thread='widget_validata_dataset',
                                 level='info')


@magicgui(call_button='Delete Dataset',
          dataset_name={'label': 'Dataset name',
                        'choices': startup_list_datasets,
                        'tooltip': f'Name of the dataset to be deleted'},
          )
def widget_delete_dataset(dataset_name: str = startup_list_datasets[0]):
    delete_dataset(dataset_name)


@widget_create_dataset.called.connect
def _on_create_dataset_called(new_dataset: dict):
    new_dataset_list = list_datasets()
    if not widget_add_stack.visible:
        widget_add_stack.show()
    widget_add_stack.dataset_name.choices = new_dataset_list
    widget_add_stack.dataset_name.value = new_dataset['name']

    if not widget_delete_dataset.visible:
        widget_delete_dataset.show()

    widget_delete_dataset.dataset_name.choices = new_dataset_list
    widget_delete_dataset.dataset_name.value = new_dataset['name']

    if not widget_validata_dataset.visible:
        widget_validata_dataset.show()

    widget_validata_dataset.dataset_name.choices = new_dataset_list
    widget_validata_dataset.dataset_name.value = new_dataset['name']


if startup_list_datasets == empty_dataset:
    widget_add_stack.hide()
    widget_delete_dataset.hide()
    widget_validata_dataset.hide()
