# Quick Start

## Pipeline Usage (Napari viewer)
PlantSeg app can also be started using napari as a viewer.
First, activate the newly created conda environment with:
```bash
conda activate plant-seg
```

then, start the plantseg in napari
```bash
$ plantseg --napari
```
A more in depth guide can be found in our [documentation (GUI)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/).

## Pipeline Usage (GUI)
PlantSeg app can also be started in a GUI mode, where basic user interface allows to configure and run the pipeline.
First, activate the newly created conda environment with:
```bash
conda activate plant-seg
```

then, run the GUI by simply typing:
```bash
$ plantseg --gui
```
A more in depth guide can be found in our [documentation (Classic GUI)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/).

## Pipeline Usage (command line)
Our pipeline is completely configuration file based and does not require any coding.

First, activate the newly created conda environment with:
```bash
conda activate plant-seg
```
then, one can just start the pipeline with
```bash
plantseg --config CONFIG_PATH
```
where `CONFIG_PATH` is the path to the YAML configuration file. See [config.yaml](examples/config.yaml) for a sample configuration
file and our [documentation (CLI)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_cli/) for a
detailed description of the parameters.

## Using LiftedMulticut segmentation

As reported in our [paper](https://elifesciences.org/articles/57613), if one has a nuclei signal imaged together with
the boundary signal, we could leverage the fact that one cell contains only one nucleus and use the `LiftedMultict`
segmentation strategy and obtain improved segmentation. This workflow is now available in all PlantSeg interfaces.
