from concurrent.futures import Future
from pathlib import Path
from typing import Tuple

from magicgui import magicgui
from napari.types import LayerDataTuple

from plantseg import PLANTSEG_MODELS_DIR
from plantseg.training.train import unet_training
from plantseg.utils import list_all_dimensionality
from plantseg.viewer.widget.predictions import ALL_DEVICES
from plantseg.viewer.widget.utils import create_layer_name, start_threading_process, return_value_if_widget


def unet_training_wrapper(dataset_dir, model_name, patch_size, dimensionality, sparse, device, **kwargs):
    """
    Wrapper to run unet_training in a thread_worker, this is needed to allow the user to select the device
    in the headless mode.
    """
    return unet_training(dataset_dir, model_name, patch_size, dimensionality, sparse, device, **kwargs)


@magicgui(call_button='Run Training',
          dataset_dir={'label': 'Path to the dataset directory',
                       'mode': 'd',
                       'tooltip': 'Select a directory containing train and val subfolders'},
          model_name={'label': 'Trained model name',
                      'tooltip': f'Model files will be saved in f{PLANTSEG_MODELS_DIR}/model_name'},
          dimensionality={'label': 'Dimensionality',
                          'tooltip': 'Dimensionality of the data (2D or 3D). ',
                          'widget_type': 'ComboBox',
                          'choices': list_all_dimensionality()},
          sparse={'label': 'Sparse',
                  'tooltip': 'If True, SPOCO spare training algorithm will be used',
                  'widget_type': 'CheckBox'},
          device={'label': 'Device',
                  'choices': ALL_DEVICES}
          )
def widget_unet_training(dataset_dir: Path = Path.home(),
                         model_name: str = 'my-model',
                         dimensionality: str = '3D',
                         patch_size: Tuple[int, int, int] = (80, 160, 160),
                         sparse: bool = False,
                         device: str = ALL_DEVICES[0]) -> Future[LayerDataTuple]:
    out_name = create_layer_name(model_name, 'training')
    step_kwargs = dict(model_name=model_name, sparse=sparse, dimensionality=dimensionality)
    return start_threading_process(unet_training_wrapper,
                                   runtime_kwargs={
                                       'dataset_dir': dataset_dir,
                                       'model_name': model_name,
                                       'patch_size': patch_size,
                                       'dimensionality': dimensionality,
                                       'sparse': sparse,
                                       'device': device
                                   },
                                   step_name='UNet training',
                                   widgets_to_update=[],
                                   input_keys=(model_name, 'training'),
                                   out_name=out_name,
                                   layer_kwarg={'name': out_name},
                                   layer_type='image',
                                   statics_kwargs=step_kwargs
                                   )


@widget_unet_training.dimensionality.changed.connect
def _on_dimensionality_changed(dimensionality: str):
    dimensionality = return_value_if_widget(dimensionality)
    if dimensionality == '2D':
        patch_size = (1, 256, 256)
    else:
        patch_size = (80, 160, 160)

    widget_unet_training.patch_size.value = patch_size
