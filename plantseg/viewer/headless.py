import multiprocessing
import time
from pathlib import Path
from typing import List, Tuple

import dask
from magicgui import magicgui

from plantseg.viewer.dag_handler import DagHandler


def _parse_input_paths(inputs, path_suffix='_path'):
    list_input_paths = [_input for _input in inputs if _input[-len(path_suffix):] == path_suffix]
    input_hints = tuple([Path for _ in list_input_paths])
    input_names = '/'.join([_input.replace(path_suffix, '') for _input in list_input_paths])
    return list_input_paths, input_names, List[Tuple[input_hints]]


def run_workflow_headless(path):
    dag = DagHandler.from_pickle(path)
    # nicely print the dag
    print(dag)
    list_input_paths, input_names, input_hints = _parse_input_paths(dag.inputs)

    @magicgui(list_inputs={'label': input_names,
                           'layout': 'vertical'},
              out_directory={'label': 'Export directory',
                             'mode': 'd',
                             'tooltip': 'Select the directory where the files will be exported'},
              num_workers={'label': '# Workers',
                           'widget_type': 'IntSlider',
                           'tooltip': 'Define the size of the gaussian smoothing kernel. '
                                      'The larger the more blurred will be the output image.',
                           'max': multiprocessing.cpu_count(), 'min': 1},
              scheduler={'label': 'Scheduler',
                         'choices': ['multiprocessing', 'threaded']
                         }
              )
    def run(list_inputs: input_hints,
            out_directory: Path = Path.home(),
            num_workers: int = 1,
            scheduler: str = 'multiprocessing'):
        dict_of_jobs = {}
        for i, _inputs in enumerate(list_inputs):
            input_dict = {_input_name: _input_path for _input_name, _input_path in zip(list_input_paths, _inputs)}
            input_dict.update({'out_stack_name': _inputs[0].stem, 'out_directory': out_directory})
            dict_of_jobs[i] = dag.get_dag(input_dict, get_type=scheduler)

        timer = time.time()
        print('Processing started')
        with dask.config.set(num_workers=num_workers):
            dask.compute(dict_of_jobs)
        print(f'Process ended in: {time.time() - timer:.2f}s')

    run.show(run=True)
