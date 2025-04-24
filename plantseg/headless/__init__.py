"""This module contains the headless workflow for PlantSeg.

To build a headless workflow, you can:
    - Register a new workflow manually using the plantseg API.
    - Run a workflow from the napari viewer and export it as a configuration file.

The headless workflow configured can be run using the `run_headless_workflow` function.
"""

from plantseg.headless.headless import (
    run_headles_workflow_from_config,
    run_headless_workflow,
)

__all__ = ["run_headles_workflow_from_config", "run_headless_workflow"]
