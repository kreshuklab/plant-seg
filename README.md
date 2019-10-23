# plant-seg
plant-seg is a tool for cell instance aware segmentation in densely packed 3D volumetric images.
The pipeline is fully described in {manuscript}, and uses a two stages segmentation strategy (Neural Network + Segmentation).
The pipeline is tooned for plant cell tissue acquired with confocal and light sheet microscopy.
Pre-trained model are provided.  

## Installation on Linux:
- First step required to use the pipeline is installing anaconda python. 
To download anaconda python open a terminal and type:
```Bash
$  wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
Then type
```bash
$ bash ./Anaconda3-2019.10-Linux-x86_64.sh
```
and follow the instructions to complete the anaconda installation. 

- Now we can download and configure the pipeline. 
```bash
$ git clone https://github.com/hci-unihd/plant-seg
$ cd plant-seg
$ git clone https://github.com/hci-unihd/pytorch-3dunet
$ conda env create -n plant-seg -f ./plant_seg.yml
```
## Pipeline Usage:
Our Pipeline is completely configuration file based and does not require any coding.
To run it is necessary to activate the correct anaconda enviroment. Type in the terminal
```bash
$ cd ~/plant_seg/
$ conda activate plant_seg
```
Now to run the pipeline simply type
```bash
$ python plantseg/plantseg.py --config ./config.yaml
```
## Guide to Custom Configuration File:
The configuration file define all operations in our pipeline all parameters and data to process.
We provide in the [example](example/README.md) directory several configuaration example for different usage.
Some key design choices:
* path key: is used to define the file to process or if multiple files the directory containing the data 
(and extension).
* propocessing key: contains a simple set of possible operation you would need to run on your data before 
the neural network. If data are ready for neural network processing delete the entire section to skip.
More details in the config.yaml
* prediction key: contains all parameters relevant for predicting the neural network. 
In the [models](plantseg/models/README.md) directory we list details of all pre-trained model we provide.
* segmentation: contains all needed to run the Multicut pipeline. All other pipelines can be runned with the same
configuration file style but due to anaconda environment incompatibilities installing a further enviroment is needed.
All instructions are in [segmentation](plantseg/segmentation/README.md) directory.
## Trubleshooting:

## Training on New Data