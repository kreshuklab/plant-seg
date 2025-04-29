# Installation

This is the installation guide for the latest PlantSeg.
Please check the installation guide for PlantSeg v1 at [PlantSeg Legacy Installation](../plantseg_legacy/installation.md).

## Download

Download the installer from [heibox](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/) and choose according to your platform.

[:material-download: Download](https://heibox.uni-heidelberg.de/d/72b4bd3ba5f14409bfee/){ .md-button .md-button--primary target="_blank"}

## Installation

The installer comes complete with its own python installation. During the installation further dependencies need to be downloaded.

!!! warning "First Start"

    The first start can be quite slow, as the machine learning models need to be downloaded.

=== "Linux and MacOs"

    Download the Installer, make the installer script executable, then run it.
    ```bash
    chmod +x PlantSeg*.sh
    ```

===  "Windows"

    Download the installer and execute it. As the installer is not signed you 
    might need to confirm the download and that you want to run the installer.
    
    Choose a installation path without spaces in it, as those can cause issues with conda packages.
    

=== "conda-forge"

    If you want to install PlantSeg without the installer, you need to have conda and git installed. (We recommend Microforge to get conda, see [installing mamba](../contributing/#install-mamba))

    ```bash
    conda create --name plant-seg plant-seg
    conda activate plant-seg
    plantseg --napari
    ```

=== "latest git version"

    To get the latest pre-release features, install PlantSeg from git. You need to have conda and git installed. (We recommend Microforge to get conda, see [installing mamba](../contributing/#install-mamba))

    ```bash
    git clone https://github.com/kreshuklab/plant-seg.git
    cd plant-seg
    conda env create -f environment.yaml
    conda activate plant-seg
    plantseg --napari
    ```

## Optional dependencies

Certain compressed TIFF files (e.g., Zlib, ZSTD, LZMA formats) require additional codecs to be processed correctly by PlantSeg. To handle such files, install the `imagecodecs` package:

```bash
conda activate plant-seg
pip install imagecodecs
```

If you plan to use SimpleITK-based watershed segmentation, you will need to install `SimpleITK` as an additional dependency:

```bash
conda activate plant-seg
pip install SimpleITK
```

## Installing PlantSeg v1

Please check the installation guide for PlantSeg v1 at [PlantSeg Legacy Installation](../plantseg_legacy/installation.md).
