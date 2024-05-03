![alt text](plantseg/legacy_gui/logo.png)

[![doc build status](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-book.yml/badge.svg)](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-book.yml)
[![package build status](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-on-conda.yml/badge.svg)](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-on-conda.yml)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/version.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/downloads.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/license.svg)](https://anaconda.org/conda-forge/plant-seg)

<!-- [![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/version.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/latest_release_date.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/downloads.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/license.svg)](https://anaconda.org/lcerrone/plantseg) -->

| Documentation                                                                                                       | Napari GUI                                                                                                                                              | Legacy GUI                                                                                                                                          | Command Line                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [![doc build status](https://img.shields.io/badge/Documentation-Home-blue)](https://hci-unihd.github.io/plant-seg/) | [![doc build status](https://img.shields.io/badge/Documentation-GUI-blue)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/) | [![doc build status](https://img.shields.io/badge/Documentation-Lecagy-blue)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_gui/) | [![doc build status](https://img.shields.io/badge/Documentation-CLI-blue)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_cli/) |

# PlantSeg  <!-- omit in toc -->

![Illustration of Pipeline](../assets/images/main_figure_nologo.png)

[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.

### Table of Contents  <!-- omit in toc -->

* [Getting Started](#getting-started)
* [Install PlantSeg](#install-plantseg)
* [Pipeline Usage (Napari viewer)](#pipeline-usage-napari-viewer)
* [Pipeline Usage (GUI)](#pipeline-usage-gui)
* [Pipeline Usage (command line)](#pipeline-usage-command-line)
* [Data Parallelism](#data-parallelism)
  * [Optional dependencies (not fully tested on Windows)](#optional-dependencies-not-fully-tested-on-windows)
* [Repository index](#repository-index)
* [Datasets](#datasets)
* [Pre-trained networks](#pre-trained-networks)
* [Training on New Data](#training-on-new-data)
* [Using LiftedMulticut segmentation](#using-liftedmulticut-segmentation)
* [Troubleshooting](#troubleshooting)
* [Tests](#tests)
* [Citation](#citation)

## Getting Started

For detailed usage checkout our [**documentation** 📖](https://hci-unihd.github.io/plant-seg/).

## Install PlantSeg

Please go to the [documentation](https://hci-unihd.github.io/plant-seg/chapters/getting_started/installation.html) for more detailed instructions. In short, we recommend using `mamba` to install PlantSeg, which is currently supported on Linux and Windows.

* GPU version, CUDA=12.x

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=12.1 pyqt plant-seg
    ```

* GPU version, CUDA=11.x

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=11.8 pyqt plant-seg
    ```

* CPU version

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch cpuonly pyqt plant-seg
    ```

The above command will create new conda environment `plant-seg` together with all required dependencies.

## Pipeline Usage (Napari viewer)

PlantSeg app can also be started using napari as a viewer.
First, activate the newly created conda environment with:

```bash
conda activate plant-seg
```

then, start the plantseg in napari

```bash
plantseg --napari
```

A more in depth guide can be found in our [documentation (GUI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_interactive_napari/).

## Pipeline Usage (GUI)

PlantSeg app can also be started in a GUI mode, where basic user interface allows to configure and run the pipeline.
First, activate the newly created conda environment with:

```bash
conda activate plant-seg
```

then, run the GUI by simply typing:

```bash
plantseg --gui
```

A more in depth guide can be found in our [documentation (Classic GUI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_gui/).

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
file and our [documentation (CLI)](https://hci-unihd.github.io/plant-seg/chapters/plantseg_classic_cli/) for a
detailed description of the parameters.

## Data Parallelism

In the headless mode (i.e. when invoked with `plantseg --config CONFIG_PATH`) the prediction step will run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.

```bash
CUDA_VISIBLE_DEVICES=0,1 plantseg --config CONFIG_PATH
```

### Optional dependencies (not fully tested on Windows)

Some types of compressed tiff files require an additional package to be read correctly (eg: Zlib,
ZSTD, LZMA, ...). To run plantseg on those stacks you need to install `imagecodecs`.
In the terminal:

```bash
conda activate plant-seg
pip install imagecodecs
```

Experimental support for SimpleITK watershed segmentation has been added to PlantSeg version 1.1.8. This features can be used only
after installing the SimpleITK package:

```bash
conda activate plant-seg
pip install SimpleITK
```

## Repository index

The PlantSeg repository is organised as follows:

* **plantseg**: Contains the source code of PlantSeg.
* **conda-reicpe**: Contains all necessary code and configuration to create the anaconda package.
* **Documentation-GUI**: Contains a more in-depth documentation of PlantSeg functionality.
* **evaluation**: Contains all script required to reproduce the quantitative evaluation in
[Wolny et al.](https://www.biorxiv.org/content/10.1101/2020.01.17.910562v1).
* **examples**: Contains the files required to test PlantSeg.
* **tests**: Contains automated tests that ensures the PlantSeg functionality are not compromised during an update.

## Datasets

We publicly release the datasets used for training the networks which available as part of the *PlantSeg* package.
Please refer to [our publication](https://www.biorxiv.org/content/10.1101/2020.01.17.910562v1) for more details about the datasets:

* *Arabidopsis thaliana* ovules dataset (raw confocal images + ground truth labels)
* *Arabidopsis thaliana* lateral root (raw light sheet images + ground truth labels)

Both datasets can be downloaded from [our OSF project](https://osf.io/uzq3w/)

## Pre-trained networks

The following pre-trained networks are provided with PlantSeg package out-of-the box and can be specified in the config file or chosen in the GUI.

* `generic_confocal_3D_unet` - alias for `confocal_3D_unet_ovules_ds2x` see below
* `generic_light_sheet_3D_unet` - alias for `lightsheet_3D_unet_root_ds1x` see below
* `confocal_3D_unet_ovules_ds1x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on original resolution, voxel size: (0.235x0.075x0.075 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds2x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on 1/2 resolution, voxel size: (0.235x0.150x0.150 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds3x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules on 1/3 resolution, voxel size: (0.235x0.225x0.225 µm^3) (ZYX) with BCEDiceLoss
* `confocal_2D_unet_ovules_ds2x` - a variant of 2D U-Net trained on confocal images of *Arabidopsis* ovules. Training the 2D U-Net is done on the Z-slices (1/2 resolution, pixel size: 0.150x0.150 µm^3) with BCEDiceLoss
* `confocal_3D_unet_ovules_nuclei_ds1x` - a variant of 3D U-Net trained on confocal images of *Arabidopsis* ovules nuclei stain on original resolution, voxel size: (0.35x0.1x0.1 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds1x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds2x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on 1/2 resolution, voxel size: (0.25x0.325x0.325 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds3x` - a variant of 3D U-Net trained on light-sheet images of *Arabidopsis* lateral root on 1/3 resolution, voxel size: (0.25x0.4875x0.4875 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_ds1x` - a variant of 2D U-Net trained on light-sheet images of *Arabidopsis* lateral root. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss
* `lightsheet_3D_unet_root_nuclei_ds1x` - a variant of 3D U-Net trained on light-sheet images *Arabidopsis* lateral root nuclei on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_nuclei_ds1x` - a variant of 2D U-Net trained on light-sheet images *Arabidopsis* lateral root nuclei. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss.
* `confocal_3D_unet_sa_meristem_cells` - a variant of 3D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. <https://doi.org/10.17863/CAM.7793>. voxel size: (0.25x0.25x0.25 µm^3) (ZYX)
* `confocal_2D_unet_sa_meristem_cells` - a variant of 2D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. <https://doi.org/10.17863/CAM.7793>.  pixel size: (25x0.25 µm^3) (YX)
* `lightsheet_3D_unet_mouse_embryo_cells` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ)
* `confocal_3D_unet_mouse_embryo_nuclei` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ)

Selecting a given network name (either in the config file or GUI) will download the network into the `~/.plantseg_models`
directory.
Detailed description of network training can be found in our [paper](#citation).

The PlantSeg home directory can be configured with the `PLANTSEG_HOME` environment variable.

```bash
export PLANTSEG_HOME="/path/to/plantseg/home"
```

## Training on New Data

For training new models we rely on the [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).
A similar configuration file can be used for training on new data and all the instructions can be found in the repo.
When the network is trained it is enough to create `~/.plantseg_models/MY_MODEL_NAME` directory
and copy the following files into it:

* configuration file used for training: `config_train.yml`
* snapshot of the best model across training: `best_checkpoint.pytorch`
* snapshot of the last model saved during training: `last_checkpoint.pytorch`

The later two files are automatically generated during training and contain all neural networks parameters.

Now you can simply use your model for prediction by setting the [model_name](examples/config.yaml) key to `MY_MODEL_NAME`.

If you want your model to be part of the open-source model zoo provided with this package, please contact us.

## Using LiftedMulticut segmentation

As reported in our [paper](https://elifesciences.org/articles/57613), if one has a nuclei signal imaged together with
the boundary signal, we could leverage the fact that one cell contains only one nucleus and use the `LiftedMultict`
segmentation strategy and obtain improved segmentation. This workflow is now available in all PlantSeg interfaces.

## Troubleshooting

See [troubleshooting](https://hci-unihd.github.io/plant-seg/chapters/getting_started/troubleshooting) for a list of common issues and their solutions.

## Tests

In order to run tests make sure that `pytest` is installed in your conda environment. You can run your tests
simply with `python -m pytest` or `pytest`. For the latter to work you need to install `plantseg` locally in "develop mode"
with `pip install -e .`.

## Citation

```
@article{wolny2020accurate,
  title={Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
  author={Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, S{\"o}ren and Wilson-S{\'a}nchez, David and Lymbouridou, Rena and others},
  journal={Elife},
  volume={9},
  pages={e57613},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```
