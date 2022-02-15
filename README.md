![alt text](./plantseg/gui/logo.png)

[![build status](https://github.com/hci-unihd/plant-seg/actions/workflows/build-deploy-on-conda.yml/badge.svg)](https://github.com/hci-unihd/plant-seg/actions/)

# PlantSeg

![alt text](./Documentation-GUI/images/main_figure_nologo.png)
[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.  

## News:
* As of version 1.4.3 plantseg is natively supported on Windows!

## Getting Started
The recommended way of installing plantseg is via the conda package, which is currently supported on Linux and Windows.
Running plantseg on other operating systems on Mac OS is currently possible only via a Docker image
that we provide  ([see below](#docker-image)).

### Prerequisites for conda package
* Linux or Windows 
* (Optional) Nvidia GPU with official Nvidia drivers installed

* Native MacOS installation (not yet M1) coming soon. 

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
Follow the instructions to complete the anaconda installation. 
The `Miniconda3-latest-Linux-x86_64.sh` file can be safely deleted.

#### Install PlantSeg using conda
The tool can be installed directly by executing in the terminal:
```bash
conda create -n plant-seg -c pytorch -c conda-forge -c lcerrone -c awolny pytorch=1.9 pytorch-3dunet=1.3.7 plantseg
```
Above command will create new conda environment `plant-seg` together with all required dependencies.

### Install on Windows
#### Install Anaconda python
First step required to use the pipeline is installing anaconda python.
If you already have a working anaconda setup you can go directly to next item. Anaconda can be downloaded for all 
platforms from here [anaconda](https://www.anaconda.com/products/individual). We suggest to use Miniconda, 
because it is lighter and install fewer unnecessary packages.

Miniconda can be downloaded from [miniconda](https://docs.conda.io/en/latest/miniconda.html). Download the 
executable `.exe` for your Windows version and follow the installation instructions.

#### Install PlantSeg using conda
The tool can be installed directly by executing in the anaconda prompt 
(***For installing and running plantseg this is equivalent to a linux terminal***):
```bash
conda create -n plant-seg -c pytorch -c conda-forge -c lcerrone -c awolny plantseg pillow=8.4
```
Above command will create new conda environment `plant-seg` together with all required dependencies.


### Optional dependencies (not fully tested on windows)
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
file and detailed description of the parameters.

### Guide to Custom Configuration File
The configuration file defines all the operations in our pipeline together with the data to be processed.
Please refer to [config.yaml](examples/config.yaml) for a sample configuration of the pipeline and detailed explanation
of all of the parameters.

Some key design choices:
* `path` attribute: is used to define either the file to process or the directory containing the data.
* `preprocessing` attribute: contains a simple set of possible operations one would need to run on their own data before calling the neural network. 
If data is ready for neural network processing either delete the entire section or set `state: False` in order to skip this step.
Detailed instructions can be found at [Data Processing](Documentation-GUI/Data-Processing.md).
* `cnn_prediction` attribute: contains all parameters relevant for predicting with neural network. 
Description of all pre-trained models provided with the package are described below.
Detailed instructions can be found at [Predictions](Documentation-GUI/Predictions.md).
* `segmentation` attribute: contains all parameters needed to run the partitioning algorithm (i.e. final segmentation). 
Detailed instructions can be found at [Segmentation](Documentation-GUI/Segmentation.md).

### Additional information

The PlantSeg related files (models, configs) will be placed inside your home directory under `~/.plantseg_models`. 

Our pipeline uses the PyTorch library for the CNN predictions. PlantSeg can be run on systems without GPU, however 
for maximum performance we recommend that the application is run on a machine with a high performance GPU for deep learning.
If `CUDA_VISIBLE_DEVICES` environment variable is not specified the prediction task will be distributed on all available GPUs.
E.g. run: `CUDA_VISIBLE_DEVICES=0 plantseg --config CONFIG_PATH` to restrict prediction to a given GPU.

## Repository index
The PlantSeg repository is organised as follows:
* **plantseg**: Contains the source code of PlantSeg.
* **conda-reicpe**: Contains all necessary code and configuration to create the anaconda package.
* **Documentation-GUI**: Contains an more in-depth documentation of PlantSeg functionality.
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

* `generic_confocal_3d_unet` - alias for `confocal_unet_bce_dice_ds2x` see below
* `generic_light_sheet_3d_unet` - alias for `lightsheet_unet_bce_dice_ds1x` see below
* `confocal_unet_bce_dice_ds1x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on original resolution, voxel size: (0.235x0.075x0.075 µm^3) (ZYX) with BCEDiceLoss
* `confocal_unet_bce_dice_ds2x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on 1/2 resolution, voxel size: (0.235x0.150x0.150 µm^3) (ZYX) with BCEDiceLoss
* `confocal_unet_bce_dice_ds3x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules on 1/3 resolution, voxel size: (0.235x0.225x0.225 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_unet_bce_dice_ds1x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on original resolution, voxel size: (0.25x0.1625x0.1625 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_unet_bce_dice_ds2x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on 1/2 resolution, voxel size: (0.25x0.325x0.325 µm^3) (ZYX) with BCEDiceLoss
* `lightsheet_unet_bce_dice_ds3x` - a variant of 3D U-Net trained on light-sheet images of _Arabidopsis_ lateral root on 1/3 resolution, voxel size: (0.25x0.4875x0.4875 µm^3) (ZYX) with BCEDiceLoss
* `confocal_2D_unet_bce_dice_ds1x` - a variant of 2D U-Net trained on confocal images of _Arabidopsis_ ovules. Training the 2D U-Net is done on the Z-slices (pixel size: 0.075x0.075 µm^3) with BCEDiceLoss
* `confocal_2D_unet_bce_dice_ds2x` - a variant of 2D U-Net trained on confocal images of _Arabidopsis_ ovules. Training the 2D U-Net is done on the Z-slices (1/2 resolution, pixel size: 0.150x0.150 µm^3) with BCEDiceLoss
* `confocal_2D_unet_bce_dice_ds3x` - a variant of 2D U-Net trained on confocal images of _Arabidopsis_ ovules. Training the 2D U-Net is done on the Z-slices (1/3 resolution, pixel size: 0.225x0.225 µm^3) with BCEDiceLoss
* `lightsheet_unet_bce_dice_nuclei_ds1x` - a variant of 3D U-Net trained on light-sheet images _Arabidopsis_ lateral root nuclei on original resolution, voxel size: (0.235x0.075x0.075 µm^3) (ZYX) with BCEDiceLoss
* `confocal_unet_bce_dice_nuclei_stain_ds1x` - a variant of 3D U-Net trained on confocal images of _Arabidopsis_ ovules nuclei stain on original resolution, voxel size: (0.35x0.1x0.1 µm^3) (ZYX) with BCEDiceLoss
* `confocal_PNAS_3d` - a variant of 3D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. https://doi.org/10.17863/CAM.7793. voxel size: (0.25x0.25x0.25 µm^3) (ZYX)
* `confocal_PNAS_2d` - a variant of 2D U-Net trained on confocal images of shoot apical meristem dataset from: Jonsson, H., Willis, L., & Refahi, Y. (2017). Research data supporting Cell size and growth regulation in the Arabidopsis thaliana apical stem cell niche. https://doi.org/10.17863/CAM.7793.  pixel size: (25x0.25 µm^3) (YX)

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
the boundary signal, we could leverage the fact that one cell contains only one nuclei and use the `LiftedMultict` 
segmentation strategy and obtain improved segmentation.
We're going to use the _Arabidopsis thaliana_ lateral root as an example. The `LiftedMulticut` strategy consist of running
PlantSeg two times:
1. Using PlantSeg to predict the nuclei probability maps using the `lightsheet_unet_bce_dice_nuclei_ds1x` network.
In this case only the pre-processing and CNN prediction steps are enabled in the config, see [example config](plantseg/resources/nuclei_predictions_example.yaml).
```bash
plantseg --config nuclei_predictions_example.yaml 
```
2. Using PlantSeg to segment the input image with `LiftedMulticut` algorithm given the nuclei probability maps from the 1st step.
See [example config](plantseg/resources/lifted_multicut_example.yaml). The notable difference is that in the `segmentation`
part of the config we set `name: LiftedMulticut` and the `nuclei_predictions_path` as the path to the directory where the nuclei pmaps
were saved in step 1. Also make sure that the `path` attribute points to the raw files containing the cell boundary staining (NOT THE NUCLEI).
```bash
plantseg --config lifted_multicut_example.yaml
```

If case when the nuclei segmentation is given, one should skip step 1., add `is_segmentation=True` flag in the [config](plantseg/resources/lifted_multicut_example.yaml)
and directly run step 2.

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
conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c awolny -c conda-forge cudatoolkit=<YOU_CUDA_VERSION> plantseg
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

* If you use plantseg from the comand line and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```

Please make sure that your configuratiuon has the correct formatting and contains all required keys. 
An updated example can be found inside the directory `examples`, in this repository.

* If when trying to execute the Lifted Multicut pipeline you receive an error like:
```
'cannot import name 'lifted_problem_from_probabilities' from 'elf.segmentation.features''
```
The solution is to re-install [elf](https://github.com/constantinpape/elf) via
```
conda install -c conda-forge -c cpape elf
```

* PlantSeg is under active development so it may happen that the models/configuration files saved in `~/.plantseg_modes`
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
