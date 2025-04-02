from pathlib import Path

import numpy as np
import yaml

from plantseg.headless.headless import run_headless_workflow
from plantseg.io.tiff import create_tiff
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks.dataprocessing_tasks import gaussian_smoothing_task
from plantseg.tasks.io_tasks import export_image_task, import_image_task
from plantseg.tasks.workflow_handler import workflow_handler


def create_random_tiff(tmpdir, name) -> Path:
    path_tiff = Path(tmpdir) / name
    create_tiff(
        path_tiff,
        np.random.rand(32, 32).astype("float32"),
        voxel_size=VoxelSize(voxels_size=(1, 1, 1), unit="um"),
        layout="YX",
    )
    return path_tiff


def test_create_workflow(tmp_path):
    # Create an empty tiff file
    path_tiff = create_random_tiff(tmp_path, "test.tiff")

    workflow_handler.clean_dag()

    ps_1 = import_image_task(
        input_path=path_tiff, key="raw", semantic_type="raw", stack_layout="YX"
    )
    ps_2 = gaussian_smoothing_task(image=ps_1, sigma=1.0)
    export_image_task(
        image=ps_2,
        export_directory=path_tiff.parent,
        name_pattern="{image_name}_export",
        scale_to_origin=True,
    )

    workflow_handler.save_to_yaml(tmp_path / "workflow.yaml")

    dag = workflow_handler.dag
    assert len(dag.list_tasks) == 3
    assert len(dag.inputs[0].keys()) == 3

    # Run the headless workflow

    path_tiff_1 = create_random_tiff(tmp_path, "test1.tiff")
    path_tiff_2 = create_random_tiff(tmp_path, "test2.tiff")

    with open(tmp_path / "workflow.yaml", "r") as file:
        config = yaml.safe_load(file)

    job_list = [
        {
            "input_path": str(path_tiff_1),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
        {
            "input_path": str(path_tiff_2),
            "export_directory": str(tmp_path / "output"),
            "name_pattern": "{file_name}_export",
        },
    ]

    config["inputs"] = job_list

    with open(tmp_path / "workflow.yaml", "w") as file:
        yaml.dump(config, file)

    run_headless_workflow(tmp_path / "workflow.yaml")

    results_dir = tmp_path / "output"
    results = list(results_dir.glob("*"))
    assert len(results) == 2, results
