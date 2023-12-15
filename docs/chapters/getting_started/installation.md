# Installation

## Prerequisites for Conda package
* Linux or Windows
* (Optional) Nvidia GPU with official Nvidia drivers installed

* Native MacOS installation (not yet M1) coming soon.

## Install on Linux
### Install Anaconda python
The first step required to use the pipeline is installing anaconda python.
You can go directly to the next item if you already have a working anaconda setup. Anaconda can be downloaded for all
platforms from here [anaconda](https://www.anaconda.com/products/individual). We suggest using Miniconda
because it is lighter and install fewer unnecessary packages.

To download Anaconda Python open a terminal and type
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Then install by typing:
```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```
Follow the instructions to complete the anaconda installation.
The `Miniconda3-latest-Linux-x86_64.sh` file can be safely deleted.

### Install PlantSeg using conda
PlantSeg can be installed directly by executing in the terminal:
```bash
conda create -n plant-seg -c pytorch -c conda-forge -c lcerrone -c awolny python=3.9 pytorch-3dunet=1.3.7 plantseg napari
```
The above command will create a new Conda environment, `plant-seg`, with all required dependencies.

## Install on Windows
### Install Anaconda python
The first step required to use the pipeline is installing anaconda python.
You can go directly to the next item if you already have a working anaconda setup. Anaconda can be downloaded for all
platforms from here [anaconda](https://www.anaconda.com/products/individual). We suggest using Miniconda
because it is lighter and install fewer unnecessary packages.

Miniconda can be downloaded from [miniconda](https://docs.conda.io/en/latest/miniconda.html). Download the
executable `.exe` for your Windows version and follow the installation instructions.

### Install PlantSeg using conda
PlantSeg can be installed directly by executing in the terminal:
```bash
conda create -n plant-seg -c pytorch -c conda-forge -c lcerrone -c awolny python=3.9 pytorch-3dunet=1.3.7 plantseg napari
```
The above command will create a new Conda environment, `plant-seg`, with all required dependencies.

## Optional dependencies (not fully tested on Windows)
Some types of compressed tiff files require an additional package to be load correctly (e.g.: Zlib,
ZSTD, LZMA, ...). To run PlantSeg on those stacks, you need to install `imagecodecs`.
In the terminal:
```bash
conda activate plant-seg
pip install imagecodecs
```

Experimental support for SimpleITK watershed segmentation has been added to PlantSeg version 1.1.8. These features can be used only
after installing the SimpleITK package:
```bash
conda activate plant-seg
pip install SimpleITK
```
