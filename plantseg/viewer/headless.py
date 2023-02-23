import multiprocessing
import time
from pathlib import Path
from typing import List, Tuple

import dask
import distributed
from magicgui import magicgui
from tqdm import tqdm

from plantseg.viewer.dag_handler import DagHandler
from plantseg.viewer.widget.predictions import ALL_DEVICES, ALL_CUDA_DEVICES

all_gpus_str = f'all {len(ALL_CUDA_DEVICES)} gpus'
ALL_GPUS = [all_gpus_str] if len(ALL_CUDA_DEVICES) > 0 else []
ALL_DEVICES_HEADLESS = ALL_DEVICES + ALL_GPUS


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
              device={'label': 'Device',
                      'choices': ALL_DEVICES_HEADLESS},
              num_workers={'label': '# Workers',
                           'widget_type': 'IntSlider',
                           'tooltip': 'Define the size of the gaussian smoothing kernel. '
                                      'The larger the more blurred will be the output image.',
                           'max': multiprocessing.cpu_count(), 'min': 1},
              scheduler={'label': 'Scheduler',
                         'choices': ['multiprocessing', 'threaded']
                         },
              call_button='Run PlantSeg'
              )
    def run(list_inputs: input_hints,
            out_directory: Path = Path.home(),
            device: str = ALL_DEVICES_HEADLESS[0],
            num_workers: int = 1,
            scheduler: str = 'multiprocessing'):
        dict_of_jobs = {}
        cluster = distributed.LocalCluster(n_workers=num_workers, threads_per_worker=1)
        client = distributed.Client(cluster)
        print(f"You can check the execution of the workflow at: \n{client.dashboard_link}\n")

        print('Setting up jobs...')
        for i, _inputs in enumerate(tqdm(list_inputs)):
            if device == all_gpus_str:
                device = ALL_DEVICES[i % len(ALL_CUDA_DEVICES)]

            input_dict = {_input_name: _input_path for _input_name, _input_path in zip(list_input_paths, _inputs)}
            input_dict.update({'out_stack_name': _inputs[0].stem, 'out_directory': out_directory, 'device': device})
            dict_of_jobs[i] = dag.get_dag(input_dict, get_type=scheduler)

        timer = time.time()
        print('Processing started...')
        results = [client.compute(job) for job in dict_of_jobs.values()]
        client.gather(results)
        print(f'Process ended in: {time.time() - timer:.2f}s')
        client.shutdown()

    run.show(run=True)
