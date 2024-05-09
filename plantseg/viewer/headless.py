import multiprocessing
import time
from pathlib import Path
from typing import List, Tuple

from dask.distributed import LocalCluster, Client
from magicgui import magicgui
from tqdm import tqdm

from plantseg.viewer.dag_handler import DagHandler
from plantseg.viewer.widget.predictions import ALL_DEVICES, ALL_CUDA_DEVICES

all_gpus_str = f'all gpus: {len(ALL_CUDA_DEVICES)}'
ALL_GPUS = [all_gpus_str] if ALL_CUDA_DEVICES else []
ALL_DEVICES_HEADLESS = ALL_DEVICES + ALL_GPUS
MAX_WORKERS = len(ALL_CUDA_DEVICES) if ALL_CUDA_DEVICES else multiprocessing.cpu_count()


def parse_input_paths(inputs: List[str], path_suffix: str = '_path') -> Tuple[List[str], str, Tuple[type, ...]]:
    input_paths = [_input for _input in inputs if _input.endswith(path_suffix)]
    input_hints = tuple([Path] * len(input_paths))
    input_names = '/'.join(input.replace(path_suffix, '') for input in input_paths)
    return input_paths, input_names, input_hints


def run_workflow_headless(path: Path):
    dag = DagHandler.from_pickle(path)
    print(dag)
    input_paths, input_names, input_hints = parse_input_paths(dag.inputs)

    @magicgui(list_inputs={'label': input_names,
                           'layout': 'vertical'},
              out_directory={'label': 'Export directory',
                             'mode': 'd',
                             'tooltip': 'Select export directory'},
              device={'label': 'Device',
                      'choices': ALL_DEVICES_HEADLESS},
              num_workers={'label': '# Workers',
                           'widget_type': 'IntSlider',
                           'tooltip': 'Set number of workers.',
                           'max': MAX_WORKERS, 'min': 1},
              scheduler={'label': 'Scheduler',
                         'choices': ['multiprocessing', 'threaded']},
              call_button='Run PlantSeg')
    def run(list_inputs: input_hints,  # FIXME: This is wrong, maybe should be List[Path]
            out_directory: Path = Path.home(),
            device: str = ALL_DEVICES_HEADLESS[0],
            num_workers: int = MAX_WORKERS,
            scheduler: str = 'multiprocessing'):
        with LocalCluster(n_workers=num_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
            print(f"Dashboard link: \n{client.dashboard_link}\n")
            print('Setting up jobs...')
            jobs = {}
            for i, inputs in enumerate(tqdm(list_inputs)):
                if device == all_gpus_str:
                    device = ALL_DEVICES[i % len(ALL_CUDA_DEVICES)]
                job_dict = dict(zip(input_paths, inputs))
                job_dict.update({
                    'out_stack_name': inputs[0].stem,
                    'out_directory': out_directory,
                    'device': device,
                })
                jobs[i] = dag.get_dag(job_dict, get_type=scheduler)

            start_time = time.time()
            print('Processing started...')
            results = client.compute(list(jobs.values()))
            client.gather(results)
            print(f'Process ended in: {time.time() - start_time:.2f}s')

    run.show(run=True)
