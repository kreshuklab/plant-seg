# plant-seg
[plant-seg](plantseg) is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline is fully described in [Wolny et al. 2019](https://link), and uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tooned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained model are provided.  

## Prerequisites
* Linux
* CUDA (Optional)

## Getting Started
### Setup on Linux:
- First step required to use the pipeline is installing anaconda python. If you already have a working anaconda setup you can go directly to next item. 
To download anaconda python open a terminal and type:
```Bash
$  wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
Then start the install by typing:
```bash
$ bash ./Anaconda3-2019.10-Linux-x86_64.sh
```
Follow the instructions to complete the anaconda installation. 

- Now we can download and configure the pipeline. 
```bash
$ git clone https://github.com/hci-unihd/plant-seg
$ cd plant-seg
$ git clone https://github.com/hci-unihd/pytorch-3dunet
$ conda env create -n plant-seg -f ./plant_seg.yml
```
The pipeline will be placed inside your home directory. If you want to use a a custom loaction please look at the
 troubleshooting for all info.
### Pipeline Usage:
Our Pipeline is completely configuration file based and does not require any coding.
To run it is necessary to activate the correct anaconda environment. (Assuming the pipeline is setup in 
the default location) Type in the terminal
```bash
$ cd ~/plant_seg/
$ conda activate plant_seg
```
Now to run the pipeline simply type
```bash
$ python plantseg/plantseg.py --config ./config.yaml
```
### Guide to Custom Configuration File:
The configuration file define all operations in our pipeline all parameters and data to process.
We provide in the [example](examples/README.md) directory several configuration example for different usage.
Some key design choices:
* path key: is used to define the file to process or if multiple files the directory containing the data 
(and extension).
* preprocessing key: contains a simple set of possible operation you would need to run on your data before 
the neural network. If data are ready for neural network processing delete the entire section to skip.
More details in the config.yaml
* prediction key: contains all parameters relevant for predicting the neural network. 
In the [models](plantseg/models/README.md) directory we list details of all pre-trained model we provide.
* segmentation: contains all needed to run the Multicut pipeline. All other pipelines can be run with the same
configuration file style but due to anaconda environment incompatibilities installing a further environment is needed.
All instructions are in [segmentation](plantseg/segmentation/README.md) directory.

## Trubleshooting:
* Import Error while predicting: This could be caused by a non standard location of the [pytorch-3dunet](https://github.com/hci-unihd/pytorch-3dunet) directory.
Please edit line 7 of [predict.py](plantseg/predictions/predict.py) with your custom path.
```python
pytorch_3dunet_default_directory = "/CUSTOM_PATH/pytorch-3dunet/"
```


## Training on New Data:
For training new models we rely on the [pytorch-3dunet](https://github.com/hci-unihd/pytorch-3dunet). 
A similar configuration file can be used for training on new data, all detailed instructions can be found  in the repo.
When the network is trained it is enough to copy the following file inside a directory:
* configuration file used for training named: config_train.yml
* model_best.torch
* model_last.torch

The later two files are automatically generated during training and contains all neural networks parameters.
The directory created have to be placed in the default location
```bash
$ ~/plant_seg/
```
Now you can simply use your model for prediction by editing the [model_name:](config.yaml) key in the prediction config file.\
If you want your model to be part of the open-source zoo of models please contact us.

## Citation:
