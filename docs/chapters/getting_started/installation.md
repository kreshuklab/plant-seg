# Installation

## Prerequisites for Conda package

* Linux, Windows, MacOS (not all features are available on MacOS)
* (Optional) Nvidia GPU with official Nvidia drivers installed for GPU acceleration

## Install Mamba
The easiest way to install PlantSeg is by using the [conda (Anaconda)](https://www.anaconda.com/) or 
[mamba (Miniforge)](https://mamba.readthedocs.io/en/latest/index.html) package manager. We recommend using `mamba` because it is faster and usually more consistent than `conda`.

=== "Linux"

    To download Miniforge open a terminal and type:

    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    Then install by typing:

    ```bash
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    and follow the installation instructions.
    Please refer to the [Miniforge repo](https://github.com/conda-forge/miniforge) for more information, troubleshooting and uninstallation instructions.
    The miniforge installation file `Miniforge3-*.sh` can be deleted now. 


=== "Windows/MacOS"
    The first step required to use the pipeline is installing mamba. The installation can be done by downloading the installer from the [Miniforge repo](https://github.com/conda-forge/miniforge). There you can find the download links for the latest version of Miniforge, troubleshooting and uninstallation instructions.

## Install PlantSeg using Mamba
PlantSeg can be installed directly by executing in the terminal (or PowerShell on Windows). For `conda` users, the command is identical, just replace `mamba` with `conda`.

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

Please refer to the [PyTorch](https://pytorch.org/get-started/locally/) website for more information on the available versions of PyTorch and the required CUDA version. The GPU version of Pytorch will also work on CPU only machines but has a much larger installation on disk.

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
