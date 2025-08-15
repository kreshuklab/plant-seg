# Quick Start

PlantSeg can be used in three different ways: interactively (using the Napari viewer), as a command line, or with a GUI. The following sections will guide you through the installation and usage of PlantSeg in each of these modes.

## Interactive PlantSeg with Napari Viewer

PlantSeg app can be started from the terminal.
After [installing PlantSeg](installation.md) using the installer, there should be a menu entry for to launch PlantSeg.

Alternatively, launch PlantSeg in the terminal. First activate your environment (by default this is the installation directory, change the command if needed) with:

```bash
conda activate ~/plantseg
```

then, start PlantSeg with the napari GUI:

```bash
plantseg --napari
```

A more in depth guide can be found in our [GUI documentation](../plantseg_interactive_napari/index.md).

## Run batch workflows

PlanSeg can perform batch jobs by running `yaml` workflow files.
You can easily generate them interactively through the napari GUI described above.
To edit a workflow file and set the correct paths use the editor:

```bash
plantseg --edit [workflow.yaml]
```

Then you can run the workflow:

```bash
plantseg --config workflow.yaml
```

[Learn more about Workflows here!](../workflow_gui/index.md)
