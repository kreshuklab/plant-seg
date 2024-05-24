# Installation

## Prerequisites for Conda package

* Linux or Windows
* (Optional) Nvidia GPU with official Nvidia drivers installed
* Native MacOS installation (not yet M1) coming soon.

## Install Mamba

Fist step is to install `mamba`, which is a faster alternative to `conda`.
If you have Anaconda/Miniconda installed, you can install Mamba in your base environment.
Otherwise we suggest to use Miniconda, because it is lighter than Anaconda and install fewer unnecessary packages.
Check the [Mamba documentation](https://mamba.readthedocs.io/en/latest/ "Mamba is officially recommended to be installed without Conda, but if you even know this you don't need to read this part of PlantSeg installation guide.") for more details.

=== "Linux"

    To download Miniconda open a terminal and type:

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```

    Then install by typing:

    ```bash
    bash ./Miniconda3-latest-Linux-x86_64.sh
    ```

    and follow the installation instructions.
    The `Miniconda3-latest-Linux-x86_64.sh` file can be deleted now.

=== "Windows"

    Miniconda can be downloaded from [miniconda](https://docs.conda.io/en/latest/miniconda.html).
    Download the executable `.exe` for your Windows version and follow the installation instructions.

When the Miniconda installation is complete, run:

```bash
conda install -c conda-forge mamba
```

## Install PlantSeg using Mamba

* GPU version, CUDA=12.x

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=12.1 pyqt plant-seg --no-channel-priority
    ```

* GPU version, CUDA=11.x

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch pytorch-cuda=11.8 pyqt plant-seg --no-channel-priority
    ```

* CPU version

    ```bash
    mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge pytorch cpuonly pyqt plant-seg --no-channel-priority
    ```

The above command will create new conda environment `plant-seg` together with all required dependencies.

## Install Newer Versions

If you want to install a specific version of PlantSeg that is not available on `conda-forge`,
you can install it from the `lcerrone` channel. For example, you can run the following command:

```bash
mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge -c lcerrone pytorch pytorch-cuda=12.1 pyqt plantseg
```

Difference between `conda-forge` and `lcerrone` channels:

* conda-forge/plant-seg:
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/version.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/downloads.svg)](https://anaconda.org/conda-forge/plant-seg)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/plant-seg/badges/license.svg)](https://anaconda.org/conda-forge/plant-seg)

* lcerrone/plantseg:
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/version.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/latest_release_date.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/downloads.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/license.svg)](https://anaconda.org/lcerrone/plantseg)

Ultimately you may download this repo and install it from source for the latest version.

## Optional dependencies

If you want to use the headless mode of PlantSeg, you need to install `dask[distributed]`:

```bash
conda activate plant-seg
mamba install dask distributed
```

Some types of compressed tiff files require an additional package to be load correctly (e.g.: Zlib, ZSTD, LZMA, ...).
To run PlantSeg on those stacks, you need to install `imagecodecs`.
In the terminal:

```bash
conda activate plant-seg
pip install imagecodecs
```

Experimental support for SimpleITK watershed segmentation has been added to PlantSeg version 1.1.8.
These features can be used only after installing the SimpleITK package:

```bash
conda activate plant-seg
pip install SimpleITK
```
