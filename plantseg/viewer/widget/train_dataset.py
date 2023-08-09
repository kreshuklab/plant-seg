from pathlib import Path

from magicgui import magicgui
from napari import Viewer

from plantseg import PLANTSEG_MODELS_DIR


@magicgui(call_button='Initialize Dataset',
          name={'label': 'Dataset name',
                'tooltip': f'Initialize an empty dataset with name model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing where the dataset will be created, '
                                  '{dataset_dir}/model_name/.'}
          )
def widget_create_dataset(viewer: Viewer, name: str = 'my-dataset', dataset_dir: Path = Path.home()):
    dataset_dir = dataset_dir / name


    dataset_dir.mkdir(parents=True, exist_ok=True)


@magicgui(call_button='Create Dataset',
          name={'label': 'Dataset name',
                'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing train and val subfolders'},
          )
def widget_print_dataset(viewer: Viewer, name: str = 'my-dataset', dataset_dir: Path = Path.home()):
    pass


@magicgui(call_button='Create Dataset',
          name={'label': 'Dataset name',
                'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing train and val subfolders'},
          )
def widget_add_stack(viewer: Viewer, name: str = 'my-dataset', dataset_dir: Path = Path.home()):
    pass


@magicgui(call_button='Delete Dataset',
          name={'label': 'Dataset name',
                'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing train and val subfolders'},
          )
def widget_delete_dataset(viewer: Viewer, name: str = 'my-dataset', dataset_dir: Path = Path.home()):
    pass
