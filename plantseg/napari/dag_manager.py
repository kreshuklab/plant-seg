import pickle

import dask
from dask.threaded import get as tget
from dask.multiprocessing import get as mpget
import warnings


class DagManager:
    def __init__(self):
        self.dask_dag = {}
        self.human_readable_dag = {}
        self.inputs = []
        self.outputs = []
        self.outputs_suffixes = []

    def __repr__(self):
        max_key_len = max([len(key) for key in self.human_readable_dag.keys()])
        out_str = 'PlantSeg Computation Graph Manager\n\n'
        out_str += f'\033[1mInputs\033[0m: {", ".join(self.inputs)} \n\n'
        out_str += '\033[1mDAG\033[0m: \n'
        for key, value in self.human_readable_dag.items():
            key += ' ' * (1 + max_key_len - len(key))
            process = "\033[92m {}\033[00m" .format(value["step-name"])
            process = f'  {key} <-- {process}('
            len_process = len(process) - 2 * len('\033[92m')

            for i, ikey in enumerate(value["step_inputs"]):
                off_set = '' if i == 0 else len_process * ' '
                end = ')\n' if i == (len(value["step_inputs"]) - 1) else ',\n'
                process += f'{off_set}{ikey}{end}'

            out_str += process
        return out_str

    def get_dag(self, inputs_dict, get_type='threaded'):
        for _input in self.inputs:
            if _input in inputs_dict:
                self.dask_dag[_input] = inputs_dict[_input]
            else:
                self.dask_dag[_input] = None
                warnings.warn(f'{_input} is not in {list(inputs_dict.keys())}, this might compromise the pipeline.')

        if get_type == 'threaded':
            get = tget
        elif get_type == 'multiprocessing':
            get = mpget
        else:
            raise ValueError('get_type must me either threaded or multiprocessing.')
        return get(self.dask_dag, self.outputs)

    def add_step(self, function, input_keys, output_key, step_name=None, step_params=None):
        step_name = step_name if step_name is not None else 'unknown step'
        step_params = step_params if step_params is not None else {}

        _inputs = [input_key for input_key in input_keys if input_key not in self.dask_dag.keys()]
        self.inputs += _inputs
        self.dask_dag[output_key] = (dask.delayed(function, name=step_name), *input_keys)

        self.dag_add_step_description(step_name, step_params, input_keys, output_key)

    def dag_add_step_description(self, step_name, step_params, input_keys, output_key):
        if isinstance(input_keys, tuple):
            input_keys = list(input_keys)

        self.human_readable_dag[output_key] = {'step-name': step_name,
                                               'step_inputs': input_keys,
                                               'step_params': step_params}

    def export_dag(self, path, outputs):
        self.outputs = outputs
        with open(path, 'wb') as f:
            pickle.dump(self, f)


dag = DagManager()
