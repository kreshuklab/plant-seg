import pickle

import dask
from dask.threaded import get


class DagManager:
    def __init__(self):
        self.dag = {}
        self.inputs = []
        self.outputs = []
        self.outputs_suffixes = []

    def __repr__(self):
        return f'inputs: {self.inputs} \n ' \
               f'dag: {self.dag} \n' \
               f'outputs: {self.outputs}'

    def get_dag(self, inputs):
        for key, _input in zip(self.inputs, inputs):
            self.dag[key] = _input
        return get(self.dag, self.outputs)

    def add_step(self, function, input_keys, output_key):
        self.inputs += [input_key for input_key in input_keys
                        if input_key not in self.dag.keys()]
        self.dag[output_key] = (dask.delayed(function), *input_keys)

    def export_dag(self, path, outputs, outputs_suffixes):
        self.outputs = outputs
        self.outputs_suffixes = outputs_suffixes

        with open(path, 'wb') as f:
            pickle.dump(self, f)


dag = DagManager()
