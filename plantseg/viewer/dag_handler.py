import pathlib
import pickle
import warnings
from functools import partial
from typing import List, Callable, Tuple, Union, Set

import dask
from dask.multiprocessing import get as mpget
from dask.threaded import get as tget

from plantseg.__version__ import __version__


class DagHandler:
    def __init__(self):
        self.plantseg_version = __version__
        self.complete_dag: dict = {}
        self.inputs: Set[str] = set()
        self.outputs: str = ''

    @classmethod
    def from_pickle(cls, workflow_path: Union[str, pathlib.Path]):
        with open(workflow_path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        max_key_len = [len(key) for key in self.complete_dag.keys()]
        max_key_len = 0 if len(max_key_len) == 0 else max(max_key_len)

        out_str = 'PlantSeg Computation Graph Handler\n\n'
        out_str += f'\033[1mInputs\033[0m: {", ".join(self.inputs)} \n\n'
        out_str += '\033[1mDAG\033[0m: \n'
        for key, value in self.complete_dag.items():
            key += ' ' * (1 + max_key_len - len(key))
            process = "\033[92m {}\033[00m".format(value["step-name"])
            process = f'  {key} <-- {process}('
            len_process = len(process) - 2 * len('\033[92m')

            for i, ikey in enumerate(value["step_inputs"]):
                off_set = '' if i == 0 else len_process * ' '
                end = ')\n' if i == (len(value["step_inputs"]) - 1) else ',\n'
                process += f'{off_set}{ikey}{end}'

            out_str += process
        return out_str

    def dag_to_daskgraph(self):
        daks_dag = {}
        for key, value in self.complete_dag.items():
            func = partial(value['step-func'], **value['static_params'])
            daks_dag[key] = (dask.delayed(func), *value['step_inputs'])
        return daks_dag

    def get_dag(self, inputs_dict: dict, outputs: List[str] = None, get_type='threaded'):
        dask_dag = self.dag_to_daskgraph()

        for _input in self.inputs:
            if _input in inputs_dict:
                dask_dag[_input] = dask.delayed(inputs_dict[_input])
            else:
                dask_dag[_input] = (None, None)
                warnings.warn(f'{_input} is not in {list(inputs_dict.keys())}, this might compromise the pipeline.')

        outputs = self.outputs if outputs is None else outputs

        if get_type == 'threaded':
            get = tget
        elif get_type == 'multiprocessing':
            get = mpget
        else:
            raise ValueError('get_type must me either threaded or multiprocessing.')
        return get(dask_dag, outputs)

    def add_step(self, function: Callable,
                 input_keys: Tuple[str, ...],
                 output_key: str,
                 static_params: dict = None,
                 step_name: str = None
                 ):

        step_name = step_name if step_name is not None else 'unknown step'
        static_params = static_params if static_params is not None else {}

        _inputs = {input_key for input_key in input_keys if input_key not in self.complete_dag.keys()}
        self.inputs = self.inputs.union(_inputs)

        if isinstance(input_keys, tuple):
            input_keys = list(input_keys)
        self.complete_dag[output_key] = {'step-name': step_name,
                                         'step-func': function,
                                         'step_inputs': input_keys,
                                         'static_params': static_params}

    def export_dag(self, path: Union[str, pathlib.Path], outputs=None):
        self.outputs = outputs
        path = pathlib.Path(path)
        assert path.suffix == '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        """
        This is a temporary solution to export the DAG in a human readable format, but does not work with the current
        implementation of the DAG.
        path = path.parent / (path.stem + '.yaml')
        human_readable_workflow = {'plantseg_version': self.plantseg_version,
                                   'DAG': self.complete_dag,
                                   'inputs': self.inputs,
                                   'outputs': self.outputs}

        with open(path, 'w') as f:
            yaml.dump(human_readable_workflow, f)
        """


dag_manager = DagHandler()
