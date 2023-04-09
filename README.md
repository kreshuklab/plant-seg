![alt text](plantseg/legacy_gui/logo.png)

[![build status](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-on-conda.yml/badge.svg)](https://github.com/hci-unihd/plant-seg/actions/)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/latest_release_date.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/downloads.svg)](https://anaconda.org/lcerrone/plantseg)
[![Anaconda-Server Badge](https://anaconda.org/lcerrone/plantseg/badges/license.svg)](https://anaconda.org/lcerrone/plantseg)
# PlantSeg

![alt text](../assets/images/main_figure_nologo.png)
[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.

### New in PlanSeg version 1.5:
* A new interactive plantseg mode using [napari](https://napari.org/stable/index.html) as a viewer!
* build-in segmentation proofreading
* New experimental headless mode
* New workflows

## Getting Started
The recommended way of installing plantseg is via the conda package, 
which is currently supported on Linux and Windows.
For detailed usage documentation checkout our [**wiki**](https://github.com/hci-unihd/plant-seg/wiki) 
[📖](https://github.com/hci-unihd/plant-seg/wiki):

<!---
Or quick test PlantSeg online using Google Colab (requires a google account)  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hci-unihd/plant-seg/blob/assets/plantseg_colab.ipynb)
--->

### Prerequisites for conda package
* Linux or Windows (Might work on MacOS but it's not tested).
* (Optional) Nvidia GPU with official Nvidia drivers installed. 

### Install on Linux
#### Install Anaconda python
First step required to use the pipeline is installing anaconda python.
If you already have a working anaconda setup you can go directly to next item. Anaconda can be downloaded for all 
platforms from here [anaconda](https://www.anaconda.com/products/individual). We suggest to use Miniconda, 
because it is lighter and install fewer unnecessary packages.

To download Anaconda Python open a terminal and type
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Then install by typing:
```bash
bash ./Miniconda3-latest-Linux-x86_64.sh
```
and follow the installation instructions.
The `Miniconda3-latest-Linux-x86_64.sh` file can be safely deleted.

#### Install PlantSeg using mamba
Fist step is to install mamba, which is an alternative to conda:
```bash
conda install -c conda-forge mamba
```
If you have a nvidia gpu, install PlantSeg using mamba:
```bash
mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge -c lcerrone plantseg pytorch-cuda=11.7
```
or if you don't have a nvidia gpu, install PlantSeg using mamba:
```bash
mamba create -n plant-seg -c pytorch -c conda-forge -c lcerrone plantseg cpuonly
```
The above command will create new conda environment `plant-seg` together with all required dependencies.

### Install on Windows
#### Install Anaconda python
First step required to use the pipeline is installing anaconda python.
If you already have a working anaconda setup you can go directly to next item. Anaconda can be downloaded for all 
platforms from here [anaconda](https://www.anaconda.com/products/individual). We suggest to use Miniconda, 
because it is lighter and install fewer unnecessary packages.

Miniconda can be downloaded from [miniconda](https://docs.conda.io/en/latest/miniconda.html). Download the 
executable `.exe` for your Windows version and follow the installation instructions.

#### Install PlantSeg using mamba
The tool can be installed directly by executing in the anaconda prompt the following commands
(***For installing and running plantseg this is equivalent to a linux terminal***). 
Fist step is to install mamba, which is an alternative to conda:
```bash
conda install -c conda-forge mamba
```
If you have a nvidia gpu, install PlantSeg using mamba:
```bash
mamba create -n plant-seg -c pytorch -c nvidia -c conda-forge -c lcerrone plantseg pytorch-cuda=11.7
```
or if you don't have a nvidia gpu, install PlantSeg using mamba:
```bash
mamba create -n plant-seg -c pytorch -c conda-forge -c lcerrone plantseg cpuonly
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
A more in depth guide can be found in our [wiki](https://github.com/hci-unihd/plant-seg/wiki/Napari).
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
A more in depth guide can be found in our [wiki](https://github.com/hci-unihd/plant-seg/wiki/PlantSeg-Classic-GUI).
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
file and our [wiki](https://github.com/hci-unihd/plant-seg/wiki/PlantSeg-Classic-CLI) for a
detailed description of the parameters.

## Data Parallelism
By default, if multiple GPUs are available the prediction step will be run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
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
We publicly release the datasets used for training the networks which available as part of the _PlantSeg_ package.
Please refer to [our publication](https://www.biorxiv.org/content/10.1101/2020.01.17.910562v1) for more details about the datasets:
- _Arabidopsis thaliana_ ovules dataset (raw confocal images + ground truth labels)
- _Arabidopsis thaliana_ lateral root (raw light sheet images + ground truth labels) 

Both datasets can be downloaded from [our OSF project](https://osf.io/uzq3w/)

## Pre-trained networks
The following pre-trained networks are provided with PlantSeg package out-of-the box and can be specified in the config file or chosen in the GUI.

* `generic_confocal_3D_unet` - alias for `confocal_3D_unet_ovules_ds2x` see below
* `generic_light_sheet_3D_unet` - alias for `lightsheet_3D_unet_root_ds1x` see below
* `confocal_3D_unet_ovules_ds1x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on original resolution, voxel size: (0.235x0.075x0.075 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds2x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on 1/2 resolution, voxel size: (0.235x0.150x0.150 µm^3) (ZYX) with BCEDiceLoss
* `confocal_3D_unet_ovules_ds3x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on 1/3 resolution, voxel size: (0.235x0.225x0.225 µm^3) (ZYX) with BCEDiceLoss
* `confocal_2D_unet_ovules_ds2x` - a variant of 2D U-Net trained on confocal images of _Arabidopsis_ ovules. Training the 2D U-Net is done on the Z-slices (1/2 resolution, pixel size: 0.150x0.150 µm^3) with BCEDiceLoss
* `confocal_3D_unet_ovules_nuclei_ds1x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules nuclei stain on original resolution, voxel size: (0.35x0.1x0.1 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds1x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds2x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on 1/2 resolution, voxel size: (0.25x0.325x0.325 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_3D_unet_root_ds3x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on 1/3 resolution, voxel size: (0.25x0.4875x0.4875 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_ds1x` - a variant of 2D U-Net trained on light-sheet images of _Arabidopsis_ lateral root. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss
* `lightsheet_3D_unet_root_nuclei_ds1x` - a variant of 3D U-Net trained on light-sheet images _Arabidopsis_ lateral root nuclei on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_2D_unet_root_nuclei_ds1x` - a variant of 2D U-Net trained on light-sheet images _Arabidopsis_ lateral root nuclei. Training the 2D U-Net is done on the Z-slices (pixel size: 0.1625x0.1625 µm^3) with BCEDiceLoss.
* `confocal_3D_unet_apical_stem_cells` - a variant of 3D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. https://doi.org/10.17863/CAM.7793. voxel size: (0.25x0.25x0.25 µm^3) (ZYX)
* `confocal_2D_unet_apical_stem_cells` - a variant of 2D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. https://doi.org/10.17863/CAM.7793.  pixel size: (25x0.25 µm^3) (YX)
* `lightsheet_3d_unet_mouse_embryo_cells` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ) 
* `confocal_3d_unet_mouse_embryo_nuclei` - A variant of 3D U-Net trained to predict the cell boundaries in live light-sheet images of ex-vivo developing mouse embryo. Voxel size: (0.2×0.2×1 µm^3) (XYZ) 

Selecting a given network name (either in the config file or GUI) will download the network into the `~/.plantseg_models`
directory.
Detailed description of network training can be found in our [paper](#citation).

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
segmentation strategy and obtain improved segmentation. This workflow can be used from our napari gui and from our 
[CLI](https://github.com/hci-unihd/plant-seg/wiki/PlantSeg-Classic-CLI/_edit#liftedmulticut-segmentation).

## Troubleshooting
* If you stumble in the following error message:
```
AssertionError:
The NVIDIA driver on your system is too old (found version xxxx).
Please update your GPU driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx
Alternatively, go to: http://pytorch.org to install
a PyTorch version that has been compiled with your version
of the CUDA driver.
```
or:
```
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```
It means that your cuda installation does not match the default in plantseg. 
You can check your current cuda version by typing in the terminal
```
cat /usr/local/cuda/version.txt
```
Then you can re-install the pytorch version compatible with your cuda by activating your `plant-seg` environment:
```
conda activate plant-seg
```
and 
```
conda install -c pytorch torchvision cudatoolkit=<YOU_CUDA_VERSION> pytorch
```
e.g. for cuda 9.2
```
conda install -c pytorch torchvision cudatoolkit=9.2 pytorch
```

Alternatively one can create the `plant-seg` environment from scratch and ensuring the correct version of cuda/pytorch, by:
```
conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c conda-forge cudatoolkit=<YOU_CUDA_VERSION> plantseg
```

* If you use plantseg from the GUI and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```
(or a similar message for any of the other keys)
It might be that the last session configuration file got corrupted or is outdated.
You should be able to solve it by removing the corrupted file `config_gui_last.yaml`.

If you have a standard installation of plantseg, you can remove it by executing on the terminal:
```
$ rm ~/.plantseg_models/configs/config_gui_last.yaml
```

* If you use plantseg from the command line, and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```

Please make sure that your configuration has the correct formatting and contains all required keys. 
An updated example can be found inside the directory `examples`, in this repository.

* If when trying to execute the Lifted Multicut pipeline you receive an error like:
```
'cannot import name 'lifted_problem_from_probabilities' from 'elf.segmentation.features''
```
The solution is to re-install [elf](https://github.com/constantinpape/elf) via
```
conda install -c conda-forge python-elf
```

* PlantSeg is under active development, so it may happen that the models/configuration files saved in `~/.plantseg_modes`
are outdated. In case of errors related to loading the configuration file, please close the PlantSeg app, 
remove `~/.plantseg_models` directory and try again.

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
