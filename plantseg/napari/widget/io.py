from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
from magicgui import magicgui
from napari.layers import Layer, Image, Labels
from napari.types import LayerDataTuple
from warnings import warn

from plantseg.dataprocessing.functional.dataprocessing import image_rescale, compute_scaling_factor
from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape, normalize_01
from plantseg.io import H5_EXTENSIONS, TIFF_EXTENSIONS, allowed_data_format
from plantseg.io.io import load_tiff, load_h5, create_tiff
from plantseg.napari.dag_manager import dag
from plantseg.napari.widget.utils import layer_properties
from napari.utils.notifications import show_info


def _check_layout_string(layout):
    n_c = 0
    for l in layout:
        if l not in ['x', 'c']:
            raise ValueError(f'letter {l} found in layout [{layout}], layout should contain only x and a single c')
        if l == 'c':
            n_c += 1

    if n_c != 1:
        raise ValueError(f'letter c found in layout {n_c} times, but should be present only once')


def _filter_channel(data, channel, layout):
    slices = []
    for i, l in enumerate(layout):
        if l == 'x':
            slices.append(slice(None, None))
        else:
            if channel > data.shape[i]:
                raise ValueError(f'image has only {data.shape[i]} channels along {layout}')
            slices.append(slice(channel, channel + 1))

    return np.squeeze(data[tuple(slices)])


def _advanced_load(path, key, channel, advanced_load=False, layer_type='image'):
    path = Path(path)
    base, ext = path.stem, path.suffix
    if ext not in allowed_data_format:
        raise ValueError(f'File extension is {ext} but should be one of {allowed_data_format}')

    if ext in H5_EXTENSIONS:
        key = key if advanced_load else None
        data, (voxel_size, _, _, voxel_size_unit) = load_h5(path, key=key)

    elif ext in TIFF_EXTENSIONS:
        channel, layout = channel
        data, (voxel_size, _, _, voxel_size_unit) = load_tiff(path)
        if advanced_load:
            assert data.ndim == len(layout)
            _check_layout_string(layout)
            data = _filter_channel(data, channel=channel, layout=layout)

    else:
        raise NotImplementedError()

    data = fix_input_shape(data)

    if layer_type == 'image':
        data = normalize_01(data)

    elif layer_type == 'labels':
        data = data.astype('uint16')

    return {'data': data,
            'voxel_size': voxel_size,
            'voxel_size_unit': voxel_size_unit
            }


def _unpack_load(loaded_dict, key):
    return loaded_dict.get(key)


@magicgui(
    call_button='Open file',
    path={'label': 'Pick a file (tiff or h5)',
          'tooltip': 'Select a file to be imported, the file can be a tiff or h5.'},
    name={'label': 'Layer Name',
          'tooltip': 'Define the name of the output layer, default is either image or label.'},
    layer_type={
        'label': 'Layer type',
        'tooltip': 'Select if the image is a normal image or a segmentation',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': ['image', 'labels']},
    advanced_load={'label': 'Advanced load a specific h5-key / tiff-channel',
                   'tooltip': 'If specified allows to select specific h5 dataset in a file,'
                              ' or specific channels in a tiff.'},
    key={'label': 'Key/layout (h5 only)',
         'tooltip': 'Key to be loaded from h5'},
    channel={'label': 'Channel/layout (tiff only)',
             'tooltip': 'Channel to select and channels layout'})
def open_file(path: Path = Path.home(),
              layer_type: str = 'image',
              name: str = '',
              advanced_load: bool = False,
              key: str = 'raw',
              channel: Tuple[int, str] = (0, 'xcxx'),
              ) -> LayerDataTuple:
    name = layer_type if name == '' else name

    # wrap load routine and add it to the dag
    step_params = {'key': key,
                   'channel': channel,
                   'advanced_load': advanced_load,
                   'layer_type': layer_type}
    _func = partial(_advanced_load,
                    **step_params)

    dag.add_step(_func,
                 input_keys=(f'{name}_path',),
                 output_key=f'_loaded_dict',
                 step_name='Load stack',
                 step_params=step_params)

    # locally unwrap the result
    load_dict = _func(path)
    data = load_dict['data']
    voxel_size = load_dict['voxel_size']
    voxel_size_unit = load_dict['voxel_size_unit']

    # add the key unwrapping to the dag
    for key, out_name in [('data', name),
                          ('voxel_size', f'{name}_voxel_size'),
                          ('voxel_size_unit', f'{name}_voxel_size_unit')]:
        step_params = {'key': key}
        dag.add_step(partial(_unpack_load, **step_params),
                     input_keys=(f'_loaded_dict',),
                     output_key=out_name,
                     step_name=f'Unpack stack: {key}',
                     step_params=step_params
                     )

    # return layer
    show_info(f'Napari - PlantSeg info: {name} correctly imported, voxel_size: {voxel_size} {voxel_size_unit}')
    layer_kwargs = layer_properties(name=name,
                                    scale=voxel_size,
                                    metadata={'original_voxel_size': voxel_size,
                                              'voxel_size_unit': voxel_size_unit,
                                              'root_name': name})
    return data, layer_kwargs, layer_type


def export_stack_as_tiff(data, name, directory,
                         voxel_size, voxel_size_unit,
                         func_scale, func_typecast, dtype, suffix):

    if func_scale is not None:
        data = func_scale(data)

    stack_name = f'{name}_{suffix}'
    directory = Path(directory)
    out_path = directory / f'{stack_name}.tiff'
    data = fix_input_shape(data)
    data = func_typecast(data, dtype)
    create_tiff(path=out_path, stack=data[...], voxel_size=voxel_size, voxel_size_unit=voxel_size_unit)
    return out_path


def _image_typecast(data, dtype):
    data = normalize_01(data)
    if dtype != 'float32':
        data = (data * np.iinfo(dtype).max)

    data = data.astype(dtype)
    return data


def _label_typecast(data, dtype):
    return data.astype(dtype)


def checkout(*args):
    for stack in args:
        stack = Path(stack)
        assert stack.is_file()


@magicgui(
    call_button='Export stack',
    images={'label': 'Layers to export',
            'layout': 'vertical',
            'tooltip': 'Select all layer to be exported, and (optional) set a suffix to append to each file name.'},
    data_type={'label': 'Data Type',
               'choices': ['float32', 'uint8', 'uint16'],
               'tooltip': 'Export datatype (uint16 for segmentation) and all others for images.'},
    directory={'label': 'Directory to export files',
               'mode': 'd',
               'tooltip': 'Select the directory where the files will be exported'},
    workflow_name={'label': 'Workflow name',
                   'tooltip': 'Name of the workflow object.'},
)
def export_stacks(images: List[Tuple[Layer, str]],
                  directory: Path = Path.home(),
                  rescale_to_original_resolution: bool = True,
                  data_type: str = 'float32',
                  workflow_name: str = 'workflow',
                  ) -> None:
    export_name = []
    for i, (image, image_suffix) in enumerate(images):
        # parse type and casting function to use
        if isinstance(image, Image):
            order = 1
            func_typecast = _image_typecast
            dtype = data_type

        elif isinstance(image, Labels):
            order = 0
            func_typecast = _image_typecast
            dtype = 'uint16'
            if data_type != 'uint16':
                warn(f"{data_type} is not a valid type for Labels, please use uint8 or uint16")
        else:
            raise ValueError(f'{type(image)} cannot be exported, please use Image layers or Labels layers')

        # parse metadata in the layer
        if rescale_to_original_resolution and 'original_voxel_size' in image.metadata.keys():
            output_resolution = image.metadata['original_voxel_size']
            input_resolution = image.scale
            scaling_factor = compute_scaling_factor(input_voxel_size=input_resolution,
                                                    output_voxel_size=output_resolution)
            func_scaling = partial(image_rescale, factor=scaling_factor, order=order)

        else:
            output_resolution = image.scale
            func_scaling = None

        if 'voxel_size_unit' in image.metadata.keys():
            voxel_size_unit = image.metadata['voxel_size_unit']
        else:
            voxel_size_unit = 'um'

        image_suffix = f'export_{i}' if image_suffix == '' else image_suffix
        step_params = {'func_scale': func_scaling,
                       'func_typecast': func_typecast,
                       'dtype': dtype,
                       'suffix': image_suffix
                       }

        func_export_tiff = partial(export_stack_as_tiff, **step_params)
        _ = func_export_tiff(data=image.data,
                             name=image.name,
                             directory=directory,
                             voxel_size=output_resolution,
                             voxel_size_unit=voxel_size_unit, )

        root_name = image.metadata['root_name']
        input_keys = (image.name,
                      'out_stack_name',
                      'out_directory',
                      f'{root_name}_voxel_size',
                      f'{root_name}_voxel_size_unit'
                      )

        _export_name = f'{image.name}_export'
        dag.add_step(func_export_tiff,
                     input_keys=input_keys,
                     output_key=_export_name,
                     step_name='Export',
                     step_params=step_params)
        export_name.append(_export_name)

        show_info(f'Napari - PlantSeg info: {image.name} correctly exported,'
                  f' voxel_size: {image.scale} {voxel_size_unit}')

    if export_name:
        final_export_check = 'final_export_check'
        dag.add_step(checkout,
                     input_keys=export_name,
                     output_key=final_export_check,
                     step_name='Checkout Execution',
                     step_params={})

        out_path = directory / f'{workflow_name}.pkl'
        dag.export_dag(out_path, final_export_check)
        show_info(f'Napari - PlantSeg info: workflow correctly exported')
