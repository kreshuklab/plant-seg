import logging
from pathlib import Path
from typing import Literal

import yaml

from plantseg.headless.basic_runner import SerialRunner
from plantseg.tasks.workflow_handler import TaskUserInput

logger = logging.getLogger(__name__)

Runners = Literal["serial"]

_implemented_runners = {'serial': SerialRunner}


def parse_input_path(user_input: TaskUserInput):
    value = user_input.value
    if value is None:
        raise ValueError("Input path must be provided.")

    elif isinstance(value, str):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Input path {path} does not exist.")

        return [path]

    elif isinstance(value, list):
        paths = [Path(p) for p in value]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Input path {path} does not exist.")

        return paths

    else:
        raise ValueError("Input path must be a string or a list of strings.")


def output_directory(user_input: TaskUserInput):
    user_input = user_input.value
    if user_input is None:
        raise ValueError("Output directory must be provided.")

    if not isinstance(user_input, str):
        raise ValueError("Output directory must be a string.")

    output_dir = Path(user_input)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def parse_generic_input(user_input: TaskUserInput):
    value = user_input.value

    if value is None:
        value = user_input.headless_default

    if value is None and user_input.user_input_required:
        raise ValueError(f"Input must be provided. {user_input}")

    return value


def parse_input_config(inputs_config: dict):
    inputs_config = {k: TaskUserInput(**v) for k, v in inputs_config.items()}

    list_input_keys = {}
    single_input_keys = {}
    has_input_path = False
    has_output_dir = False

    for key, value in inputs_config.items():
        if key.find("input_path") != -1:
            list_input_keys[key] = parse_input_path(value)
            has_input_path = True

        elif key.find("output_dir") != -1:
            single_input_keys[key] = output_directory(value)
            has_output_dir = True

        else:
            single_input_keys[key] = parse_generic_input(value)

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
        logger.info(f"Submitting job with input: {job_input}")
        runner.submit_job(job_input)

    logger.info("All jobs have been submitted. Running the workflow...")
    # TODO: When parallel runners are implemented, hew we need to add something to wait for all jobs to finish
