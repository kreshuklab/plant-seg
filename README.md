![alt text](./plantseg/gui/logo.png)
# PlantSeg
[PlantSeg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tuned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained models are provided.  

## Getting Started
### Prerequisites
* Linux
* Nvidia GPU + CUDA (Optional)

### Install Anaconda python
- First step required to use the pipeline is installing anaconda python.
 If you already have a working anaconda setup you can go directly to next item. 
To download Anaconda Python open a terminal and type:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
Then install by typing:
```bash
bash ./Anaconda3-2019.10-Linux-x86_64.sh
```
Follow the instructions to complete the anaconda installation.

### Install PlantSeg using conda
The tool can be installed directly by executing in the terminal:
```bash
conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c awolny -c conda-forge plantseg=1.0.4
```
Above command will create new conda environment `plant-seg` together with all required dependencies.

### Pipeline Usage (command line)
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

### Pipeline Usage (GUI)
PlantSeg app can also be started in a GUI mode, where basic user interface allows to configure and run the pipeline.
First, activate the newly created conda environment with:
```bash
conda activate plant-seg
```

then, run the GUI by simply typing:
```bash
$ plantseg --gui
```

### Guide to Custom Configuration File
The configuration file defines all the operations in our pipeline together with the data to be processed.
Please refer to [config.yaml](examples/config.yaml) for a sample configuration of the pipeline and detailed explanation
of all of the parameters.

Some key design choices:
* `path` attribute: is used to define either the file to process or the directory containing the data.
* `preprocessing` attribute: contains a simple set of possible operations one would need to run on their own data before calling the neural network. 
If data is ready for neural network processing either delete the entire section or set `state: False` in order to skip this step.
* `cnn_prediction` attribute: contains all parameters relevant for predicting with neural network. 
Description of all pre-trained models provided with the package are described below.
* `segmentation` attribute: contains all parameters needed to run the partitioning algorithm (i.e. final segmentation). 
Detailed instructions can be found in [segmentation](plantseg/segmentation/README.md) directory.

### Additional information

The PlantSeg related files (models, configs) will be placed inside your home directory under `~/.plantseg_models`. 

Our pipeline uses the PyTorch library for the CNN predictions. PlantSeg can be run on systems without GPU, however 
for maximum performance we recommend that the application is run on a machine with a high performance GPU for deep learning.
If `CUDA_VISIBLE_DEVICES` environment variable is not specified the prediction task will be distributed on all available GPUs.
E.g. run: `CUDA_VISIBLE_DEVICES=0 plantseg --config CONFIG_PATH` to restrict prediction to a given GPU.

## Docker image
We also provide a docker image with plantseg package installed.
Make sure that [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is installed on the docker host.

**Linux only**: In oder to execute the docker image in the GUI mode, fist we need to allow everyone to access X server
on the docker host. This can be done by invoking the following command in the terminal:
```bash
xhost +

```
The just run:
```
docker run -it --rm -v PATH_TO_DATASET:/root/datasets -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY wolny/plantseg

```
this will start the _PlantSeg_ GUI application. `PATH_TO_DATASET` is the directory on the docker host where the data to be processed are stored.

## Datasets
We publicly release the datasets used for training the networks which available as part of the _PlantSeg_ package.
Please refer to [our publication](https://www.biorxiv.org/content/10.1101/2020.01.17.910562v1) for more details about the datasets:
- _Arabidopsis thaliana_ ovules dataset (raw confocal images + ground truth labels) can be downloaded from [here](https://oc.embl.de/index.php/s/yUl0GGCYDfxxVVm)
- _Arabidopsis thaliana_ lateral root (raw light sheet images + ground truth labels) can be downloaded from [here](https://oc.embl.de/index.php/s/gNXDHhOS4GwBlZT)

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

* PlantSeg is under active development so it may happen that the models/configuration files saved in `~/.plantseg_modes`
are outdated. In case of errors related to loading the configuration file, please close the PlantSeg app, 
remove `~/.plantseg_models` directory and try again.


## Tests
In order to run tests make sure that `pytest` is installed in your conda environment. You can run your tests 
simply with `python -m pytest` or `pytest`. For the latter to work you need to install `plantseg` locally in "develop mode"
with `pip install -e .`. 

## Citation
```
@article {Wolny2020.01.17.910562,
	author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro,
              Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Steigleder, Susanne and Pape, 
              Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George and Lohmann,
              Jan U. and Hamprecht, Fred A. and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
	title = {Accurate And Versatile 3D Segmentation Of Plant Tissues At Cellular Resolution},
	elocation-id = {2020.01.17.910562},
	year = {2020},
	doi = {10.1101/2020.01.17.910562},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562}, 
	eprint = {https://www.biorxiv.org/content/early/2020/01/18/2020.01.17.910562.full.pdf},
	journal = {bioRxiv}
}
```
