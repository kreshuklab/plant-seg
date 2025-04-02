import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from plantseg.headless.basic_runner import SerialRunner
from plantseg.io import allowed_data_format
from plantseg.tasks.workflow_handler import RunTimeInputSchema

logger = logging.getLogger(__name__)

Runners = Literal["serial"]

_implemented_runners = {"serial": SerialRunner}


def validate_config(config: dict):
    if "inputs" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'inputs' section."
        )

    if "infos" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'infos' section."
        )

    if "list_tasks" not in config:
        raise ValueError(
            "The workflow configuration does not contain an 'list_tasks' section."
        )

    if "runner" not in config:
        logger.warning(
            "The workflow configuration does not contain a 'runner' section. Using the default serial runner."
        )
        config["runner"] = "serial"

    return config


def parse_import_image_task(input_path, allow_dir: bool) -> list[Path]:
    if isinstance(input_path, str):
        input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} does not exist.")

    if input_path.is_file():
        list_files = [input_path]
    elif input_path.is_dir():
        if not allow_dir:
            raise ValueError(
                f"Directory {input_path} is not allowed when multiple input files are expected."
            )

        list_files = list(input_path.glob("*"))
    else:
        raise ValueError(f"Path {input_path} is not a file or a directory.")

    list_files = [f for f in list_files if f.suffix.lower() in allowed_data_format]
    if not list_files:
        raise ValueError(f"No valid files found in {input_path}.")

    return list_files


def collect_jobs_list(
    inputs: dict | list[dict], inputs_schema: dict[str, RunTimeInputSchema]
) -> list[dict[str, Any]]:
    """
    Parse the inputs and create a list of jobs to run.
    """

    if isinstance(inputs, dict):
        inputs = [inputs]

    num_is_input_file = sum(
        [1 for schema in inputs_schema.values() if schema.is_input_file]
    )

    if num_is_input_file == 0:
        raise ValueError(
            "No input files found in the inputs schema. The workflow cannot run."
        )
    elif num_is_input_file > 1:
        allow_dir = False
    else:
        allow_dir = True

    all_jobs = []
    for input_dict in inputs:
        if not isinstance(input_dict, dict):
            raise ValueError(f"Input {input_dict} should be a dictionary.")

        inputs_files = {}
        for name, schema in inputs_schema.items():
            if schema.is_input_file:
                if name not in inputs_files:
                    inputs_files[name] = []

                inputs_files[name].extend(
                    parse_import_image_task(input_dict[name], allow_dir=allow_dir)
                )

        list_len = [len(files) for files in inputs_files.values()]
        if len(set(list_len)) != 1:
            raise ValueError(
                f"Inputs have different number of files. found {inputs_files}"
            )

        list_files = list(zip(*inputs_files.values()))
        list_keys = list(inputs_files.keys())
        list_jobs = [dict(zip(list_keys, files)) for files in list_files]

        for job in list_jobs:
            for key, value in input_dict.items():
                if key not in job:
                    job[key] = value
            all_jobs.append(job)
    return all_jobs


def run_headles_workflow_from_config(config: dict, path: str | Path):
    config = validate_config(config)

    inputs = config["inputs"]
    inputs_schema = config["infos"]["inputs_schema"]
    inputs_schema = {k: RunTimeInputSchema(**v) for k, v in inputs_schema.items()}

    jobs_list = collect_jobs_list(inputs, inputs_schema)

    runner = config.get("runner")
    if runner not in _implemented_runners:
        raise ValueError(f"Runner {runner} is not implemented.")

    runner = _implemented_runners[runner](path)

    for job_input in jobs_list:
        logger.info(f"Submitting job with input: {job_input}")
        runner.submit_job(job_input)

    logger.info("All jobs have been submitted. Running the workflow...")


def run_headless_workflow(path: str | Path):
    """
    Run a workflow in headless mode using the provided configuration file.

    Args:
        path (str | Path): Path to the workflow configuration file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Workflow configuration file {path} does not exist.")

    with path.open("r") as file:
        config = yaml.safe_load(file)

    run_headles_workflow_from_config(config, path=path)
