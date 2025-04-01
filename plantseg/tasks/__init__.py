import importlib
import pkgutil

from plantseg.tasks.workflow_handler import WorkflowHandler, task_tracker

__all__ = ["WorkflowHandler", "task_tracker"]


# Automatically import all functions from all submodules
# This is necessary to automatically register all tasks in the workflow handler
# Task in the plantseg.tasks module are automatically registered in the workflow handler
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals().update(
        {
            name: getattr(module, name)
            for name in dir(module)
            if callable(getattr(module, name))
        }
    )
