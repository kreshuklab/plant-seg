# Quick Start

PlantSeg can be used in three different ways: interactively (using the Napari viewer), as a command line, or with a GUI. The following sections will guide you through the installation and usage of PlantSeg in each of these modes.

## Interactive PlantSeg with Napari Viewer

PlantSeg app can be started from the terminal.
First, activate the newly created conda environment with:

```bash
mamba activate plant-seg
```

then, start the plantseg in napari

```bash
plantseg --napari
```

A more in depth guide can be found in our [documentation (GUI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/).

## Command Line PlantSeg

PlantSeg can be configured using `YAML` config files.

First, activate the newly created conda environment with:

```bash
mamba activate plant-seg
```

then, one can just start the pipeline with

```bash
plantseg --config CONFIG_PATH
```

where `CONFIG_PATH` is the path to the `YAML` configuration file. See [config.yaml](https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml) for a sample configuration
file and our [documentation (CLI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_cli/) for a
detailed description of the parameters.

## PlantSeg with Legacy GUI

!!! failure "Deprecated"
    This interface is deprecated and has been removed from PlantSeg v2. Please use the Napari viewer or the command line interface instead, or [install PlantSeg v1](../plantseg_legacy/installation.md).

PlantSeg app can also be started in a GUI mode, where basic user interface allows to configure and run the pipeline.
First, activate the newly created conda environment with:

```bash
mamba activate plant-seg
```

then, run the GUI by simply typing:

```bash
plantseg --gui
```

A more in depth guide can be found in our [documentation (Classic GUI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_gui/).
