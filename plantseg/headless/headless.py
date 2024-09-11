import logging
from pathlib import Path
from typing import Literal

import yaml

from plantseg.headless.basic_runner import SerialRunner

logger = logging.getLogger(__name__)

Runners = Literal["serial"]

_implemented_runners = {'serial': SerialRunner}


def parse_input_path(input_path: str | list[str]):
    if isinstance(input_path, str):
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path {path} does not exist.")

        elif path.is_file():
            paths = [path]

        elif path.is_dir():
            paths = list(path.glob("*"))

    elif isinstance(input_path, list):
        paths = [Path(p) for p in input_path]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Input path {path} does not exist.")

    else:
        raise ValueError("Input path must be a string or a list of strings.")

    return paths


def output_directory(output_dir: str):
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def parse_input_config(inputs_config: dict):
    list_input_keys = {}
    single_input_keys = {}
    has_input_path = False
    has_output_dir = False

    for key in inputs_config:
        if key.find("input_path") != -1:
            list_input_keys[key] = parse_input_path(inputs_config[key])
            has_input_path = True

        elif key.find("output_dir") != -1:
            single_input_keys[key] = output_directory(inputs_config[key])
            has_output_dir = True

        else:
            single_input_keys[key] = inputs_config[key]

    if not has_input_path:
        raise ValueError("The provided workflow configuration does not contain an input path.")

    if not has_output_dir:
        raise ValueError("The provided workflow configuration does not contain an output directory.")

    all_length = [len(v) for v in list_input_keys.values()]
    # check if all input paths have the same length
    if not all([_l == all_length[0] for _l in all_length]):
        raise ValueError("All input paths must have the same length.")

    num_inputs = all_length[0]

    jobs_inputs = []
    for i in range(num_inputs):
        job_input = {}
        for key in list_input_keys:
            job_input[key] = list_input_keys[key][i]

        for key in single_input_keys:
            job_input[key] = single_input_keys[key]

        jobs_inputs.append(job_input)

    return jobs_inputs


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

    job_inputs = parse_input_config(config["inputs"])

    runner = config.get("runner", "serial")
    if runner not in _implemented_runners:
        raise ValueError(f"Runner {runner} is not implemented.")

    runner = _implemented_runners[runner](path)

    for job_input in job_inputs:
        runner.run(job_input)
