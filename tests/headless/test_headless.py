from pathlib import Path

import numpy as np
import pytest
import yaml

from plantseg.core.voxelsize import VoxelSize
from plantseg.headless.headless import run_headless_workflow
from plantseg.io import create_tiff
from plantseg.tasks.dataprocessing_tasks import gaussian_smoothing_task
from plantseg.tasks.io_tasks import export_image_task, import_image_task
from plantseg.tasks.workflow_handler import workflow_handler


def create_random_tiff(tmpdir, name) -> Path:
    path_tiff = Path(tmpdir) / name
    create_tiff(
        path_tiff,
        np.random.rand(32, 32).astype('float32'),
        voxel_size=VoxelSize(voxels_size=(1, 1, 1), unit='um'),
        layout="YX",
    )
    return path_tiff


def test_create_workflow(tmp_path):
    # Create an empty tiff file
    path_tiff = create_random_tiff(tmp_path, 'test.tiff')

    workflow_handler.clean_dag()

    ps_1 = import_image_task(input_path=path_tiff, key='raw', semantic_type='raw', stack_layout='YX')
    ps_2 = gaussian_smoothing_task(image=ps_1, sigma=1.0)
    export_image_task(
        image=ps_2,
        output_directory=path_tiff.parent,
        output_file_name=None,
        custom_key_suffix='_smoothed',
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / 'workflow.yaml')

    dag = workflow_handler.dag
    assert len(dag.list_tasks) == 3
    assert len(dag.inputs.keys()) == 3

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, 'test1.tiff')
    path_tiff_2 = create_random_tiff(tmp_path, 'test2.tiff')

    with open(tmp_path / 'workflow.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['inputs']['input_path']['value'] = [str(path_tiff_1), str(path_tiff_2)]
    config['inputs']['output_directory']['value'] = str(tmp_path / 'output')

    with open(tmp_path / 'workflow.yaml', 'w') as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / 'workflow.yaml')

    results_dir = tmp_path / 'output'
    results = list(results_dir.glob('*'))
    assert len(results) == 2, results
