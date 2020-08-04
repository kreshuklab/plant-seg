import glob
import json
import os

import yaml
from flask import Flask, send_from_directory, jsonify

from plantseg import model_zoo_path
from plantseg.__version__ import __version__
from plantseg.pipeline.executor import PipelineExecutor
from plantseg.pipeline.utils import get_logger

app = Flask(__name__)

logger = get_logger('PlantsegServer')

# list of tasks
tasks = {}

# pipeline executor
pipeline_executor = PipelineExecutor(max_workers=1, max_size=1)


def load_tasks(datadir):
    """
    Loads executed task from the data directory and populates the `tasks` global variable.
    Each task is represented by two files:
    - <TASK_ID>.yaml (plantseg yaml configuration for the task)
    - <TASK_ID>.status (contains additional task attributes, e.g. status: 'pending'/'executed', duration, labels, etc)
    This way the configs created by the Web UI can later by simply executed from command line with `plantseg --config <TASK_ID>.yaml` if needed.
    It assumes that the task status and configuration filesare stored in the 'datadir/tasks'.
    This function should be executed before starting the Flask app.
    """

    global tasks
    tasks_dir = os.path.join(datadir, 'tasks')
    if not os.path.exists(tasks_dir):
        # just create a task dir if it doesn't exist
        os.makedirs(tasks_dir)
        logger.info(f'Created task directory: {tasks_dir}')
    else:
        for task_status_file in glob.glob(os.path.join(tasks_dir, '*.status')):
            # task status are stored in the files of the form <TASK_ID>.status
            task_id = int(os.path.split(task_status_file)[1].split('.')[0])
            # load task attributes from the <TASK_ID>.status file. Assumes it's a JSON format
            task_attributes = json.load(open(task_status_file, 'r'))
            task_config_file = os.path.join(tasks_dir, f'{task_id}.yaml')
            if not os.path.exists(task_config_file):
                logger.warning(f'Cannot find task configuration file corresponding to {task_status_file}. Skipping')
                continue

            task_config = yaml.safe_load(open(task_config_file, 'r'))

            # create task object
            task_object = {
                "attributes": task_attributes,
                "config": task_config
            }

            # store in task repository
            tasks[task_id] = task_object
        logger.info(f'Loaded {len(tasks)} task objects from {tasks_dir}')


def run_server(datadir, port=8070, debug=False):
    global DATA_DIR
    DATA_DIR = datadir
    assert datadir is not None
    # load a list of competed task from the DATA_DIR
    load_tasks(datadir)
    # start the flask app
    logger.info(f'Starting PlantsegServer on port: {port}. Datadir: {datadir}')
    app.run(host='0.0.0.0', port=port, debug=debug)


@app.route("/", methods=["GET"])
def root():
    # our shiny react landing page
    # TODO: show a list of tasks in the landing page, which has a button to create new task/config
    return send_from_directory(os.path.split(__file__)[0], "index.html")


@app.route("/tasks", methods=["GET"])
def get_task_list():
    """
    Returns a list of task ids
    """
    load_tasks(DATA_DIR) # reload all tasks

    output = {
        'result': list(tasks.keys())
    }

    return jsonify(output)


@app.route("/tasks/<int:task_id>", methods=["GET"])
def get_task_object(task_id):
    """
    Return task info dictionary
    """
    return tasks[task_id]


@app.route("/tasks", methods=["POST"])
def create_task(task):
    """
    Save the task instance and start the plantseg computation.
    Task instance is created via the Web UI wizard.
    Returns an id of newly created task.
    """
    global DATA_DIR
    # generate a new task_id
    task_id = _new_task_id()
    # save the task object
    tasks[task_id] = task
    # save task_id.status file
    task_status_file = os.path.join(DATA_DIR, f'{task_id}.status')
    json.dump(task['attributes'], open(task_status_file, 'w'))
    # save task config to task_id.yaml
    task_config_file = os.path.join(DATA_DIR, f'{task_id}.yaml')
    yaml.safe_dump(task['config'], open(task_config_file, 'w'))

    # initiate pipeline execution in a separate thread
    pipeline_executor.submit(task['config'])
    # FIXME: how do we change the <TASK_ID>.status when the pipeline gets executed?
    return task_id


@app.route("/info", methods=["GET"])
def get_server_info():
    """
    Returns general info about the plantseg server.
    """
    global DATA_DIR
    info = {
        'plantseg_version': __version__,
        'datadir': DATA_DIR
    }

    return info


@app.route("/datafiles", methods=["GET"])
def get_data_files():
    """
    Traverses the DATA_DIR recursively and returns a list of tiff and h5 files that are available for processing.
    All subdirectories inside DATA_DIR containing tiff/h5 files should be included in the results as well in case
    the user wants to process the whole directory using plantseg.
    This endpoint should be used to populate the drop-down field where the user selects file/directory to process.
    """
    # TODO: implement
    return []


@app.route("/models", methods=["GET"])
def get_model_names():
    """
    Returns a list of model names available in plantseg.
    """
    # TODO: we should also support uploading the model file via the server; models uploaded manually should also be included in the resulting list
    models = yaml.safe_load(open(model_zoo_path, 'r'))
    return list(models.keys())


def _new_task_id():
    if not tasks:
        # return 1 if tasks dictionary is empty
        return 1
    else:
        return max(tasks.keys()) + 1
