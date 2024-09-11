import logging
from pathlib import Path

from plantseg.tasks.workflow_handler import DAG, Task, WorkflowHandler

logger = logging.getLogger(__name__)


class Headless:
    def __init__(self, dag_path: str | Path):
        pass
