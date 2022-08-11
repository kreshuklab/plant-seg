from concurrent.futures import Future
from functools import partial
from typing import Callable, Tuple

from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

from plantseg.napari.dag_manager import dag


def identity(x):
    return x


def start_threading_process(func: Callable,
                            func_kwargs: dict,
                            out_name: str,
                            input_keys: Tuple[str, ...],
                            layer_kwarg: dict,
                            layer_type='image',
                            skip_dag=False) -> Future:
    thread_func = thread_worker(partial(func, **func_kwargs))
    future = Future()

    def on_done(result):
        show_info(f'Napari - PlantSeg info: widget computation complete')
        _func = func if not skip_dag else identity
        dag.add_step(_func, input_keys=input_keys, output_key=out_name)
        result = result, layer_kwarg, layer_type
        future.set_result(result)

    worker = thread_func()
    worker.returned.connect(on_done)
    worker.start()
    return future


def layer_properties(name, scale):
    return {'name': name, 'scale': scale}


def _find_version(old_suffix, new_suffix):
    s_idx = old_suffix.find(new_suffix)
    if s_idx != -1:
        v_idx = s_idx + len(new_suffix)
        old_suffix, current_version = old_suffix[:v_idx], old_suffix[v_idx:]

        current_version = 0 if current_version == '' else int(current_version[1:-1])
        current_version += 1
        new_version = f'[{current_version}]'
        return old_suffix, new_version

    return f'{old_suffix}_{new_suffix}', ''


def build_nice_name(base, new_suffix):
    if base.find('_') == -1:
        return f'{base}_{new_suffix}'

    *base_without_suffix, old_suffix = base.split('_')
    base_without_suffix = '_'.join(base_without_suffix)

    new_suffix, version = _find_version(old_suffix, new_suffix)
    return f'{base_without_suffix}_{new_suffix}{version}'
