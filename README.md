# PlantSeg  <!-- omit in toc -->

![alt text](docs/logos/logo.png)

[![doc build status](https://github.com/kreshuklab/plant-seg/actions/workflows/build-deploy-book.yml/badge.svg)](https://github.com/kreshuklab/plant-seg/actions/workflows/build-deploy-book.yml)
[![package build status](https://github.com/kreshuklab/plant-seg/actions/workflows/build-deploy-on-conda.yml/badge.svg)](https://github.com/kreshuklab/plant-seg/actions/workflows/build-deploy-on-conda.yml)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/version.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/downloads.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/license.svg)](https://anaconda.org/conda-forge/plant-seg)

![Illustration of Pipeline](../assets/images/main_figure_nologo.png)

[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.

## Table of Contents  <!-- omit in toc -->

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Repository Index](#repository-index)
- [Citation](#citation)

## Getting Started

For detailed usage checkout our [**documentation** 📖](https://kreshuklab.github.io/plant-seg/).

| Documentation                                                                                                       | Napari GUI                                                                                                                                              | Legacy GUI                                                                                                                                          | Command Line                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [![doc build status](https://img.shields.io/badge/Documentation-Home-blue)](https://kreshuklab.github.io/plant-seg/) | [![doc build status](https://img.shields.io/badge/Documentation-GUI-blue)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/) | [![doc build status](https://img.shields.io/badge/Documentation-Lecagy-blue)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/) | [![doc build status](https://img.shields.io/badge/Documentation-CLI-blue)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_cli/) |

## Installation

Please go to the [documentation](https://kreshuklab.github.io/plant-seg/chapters/getting_started/installation.html) for more detailed instructions. In short, we recommend using `mamba` to install PlantSeg, which is currently supported on Linux and Windows.

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

## Repository Index

The PlantSeg repository is organised as follows:

* **plantseg**: Contains the source code of PlantSeg.
* **conda-reicpe**: Contains all necessary code and configuration to create the anaconda package.
* **Documentation-GUI**: Contains a more in-depth documentation of PlantSeg functionality.
* **evaluation**: Contains all script required to reproduce the quantitative evaluation in
[Wolny et al.](https://doi.org/10.7554/eLife.57613).
* **examples**: Contains the files required to test PlantSeg.
* **tests**: Contains automated tests that ensures the PlantSeg functionality are not compromised during an update.

## Citation

```text
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
