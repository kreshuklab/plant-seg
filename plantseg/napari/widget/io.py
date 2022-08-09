from functools import partial
from pathlib import Path
from typing import List, Tuple

from magicgui import magicgui
from napari.layers import Layer, Image, Labels
from napari.types import LayerDataTuple
from warnings import warn

from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape, normalize_01
from plantseg.io import H5_EXTENSIONS, TIFF_EXTENSIONS, allowed_data_format
from plantseg.io.io import load_tiff, load_h5, create_tiff
from plantseg.napari.dag_manager import dag
from plantseg.napari.widget.utils import layer_properties


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

    return data[tuple(slices)]


def _advanced_load(path, key, channel, advanced_load=False, layer_type='image', headless=False):
    base, ext = path.stem, path.suffix
    if ext not in allowed_data_format:
        raise ValueError(f'File extension is {ext} but should be one of {allowed_data_format}')

    if ext in H5_EXTENSIONS:
        key = key if advanced_load else None
        data, (voxel_size, _, _) = load_h5(path, key=key)

    elif ext in TIFF_EXTENSIONS:
        channel, layout = channel
        data, (voxel_size, _, _) = load_tiff(path)
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

    if headless:
        return data
    return data, voxel_size


@magicgui(
    call_button='Open file',
    path={'label': 'Pick a file (tiff or h5)'},
    name={'label': 'Layer Name'},
    layer_type={
        'label': 'Layer type',
        'widget_type': 'RadioButtons',
        'orientation': 'horizontal',
        'choices': ['image', 'labels']},
    advanced_load={'label': 'Advanced load a specific h5-key / tiff-channel'},
    key={'label': 'key/layout (h5 only)'},
    channel={'label': 'channel/layout (tiff only)'})
def open_file(path: Path = Path.home(),
              layer_type: str = 'image',
              name: str = '',
              advanced_load: bool = False,
              key: str = 'raw',
              channel: Tuple[int, str] = (0, 'xcxx'),
              ) -> LayerDataTuple:
    name = layer_type if name == '' else name
    _func_gui = partial(_advanced_load,
                        key=key,
                        channel=channel,
                        advanced_load=advanced_load,
                        layer_type=layer_type,
                        headless=False)
    _func_dask = partial(_advanced_load,
                         key=key,
                         channel=channel,
                         advanced_load=advanced_load,
                         layer_type=layer_type,
                         headless=True)

    data, voxel_size = _func_gui(path)
    dag.add_step(_func_dask, input_keys=(f'{name}_path',), output_key=name)
    return data, layer_properties(name=name, scale=voxel_size), layer_type


def export_stack_as_tiff(data, name, voxel_size, directory, dtype, suffix):
    stack_name = f'{name}_{suffix}'
    out_path = directory / f'{stack_name}.tiff'
    data = fix_input_shape(data)
    data = data.astype(dtype)
    create_tiff(path=out_path, stack=data[...], voxel_size=voxel_size)
    return stack_name


@magicgui(
    call_button='Export stack',
    images={'label': 'Layers to export', 'layout': 'vertical'},
    data_type={'label': 'Data Type', 'choices': ['float32', 'uint8', 'uint16']},
    directory={'label': 'Directory to export files'})
def export_stacks(images: List[Tuple[Layer, str]],
                  directory: Path = Path.home(),
                  data_type: str = 'float32',
                  ) -> None:
    names, suffixes = [], []
    for image, image_suffix in images:
        if isinstance(image, Image):
            dtype = data_type

        elif isinstance(image, Labels):
            if data_type in ['uint8', 'uint16']:
                dtype = data_type
            else:
                dtype = 'uint16'
                warn(f"{data_type} is not a valid type for Labels, please use uint8 or uint16")
        else:
            raise ValueError(f'{type(image)} cannot be exported, please use Image layers or Labels layers')

        _ = export_stack_as_tiff(data=image.data,
                                 name=image.name,
                                 voxel_size=image.scale,
                                 directory=directory,
                                 dtype=dtype,
                                 suffix=image_suffix)
        names.append(image.name)
        suffixes.append(image_suffix)

    out_path = directory / 'workflow.pkl'
    dag.export_dag(out_path, names, suffixes)
