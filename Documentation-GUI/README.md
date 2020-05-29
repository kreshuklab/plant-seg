
# PlantSeg introduction

PlantSeg is a tool for 3D and 2D segmentation. The tools is fundamentally composed of two main steps. 

* ***Cell boundary predictions***: Where a convolutional neural network is used to extract a 
voxel wise boundary classification. The neural network is capable of filtering out very different types/intensity of 
noise, homogenising the signal strength and fixing imaging defect (such as missing cell boundary detection).

* ***Segmentation as partitioning***: The output of the fist step can be used directly for automated segmentation. 
We implemented 4 different algorithm for segmentation, each with peculiar features. This type of approach is expecially 
well suited for segmentation of densely packed cell.

For a complete description of the methods used please check out our [manuscript](https://www.biorxiv.org/content/10.1101/2020.01.17.910562v1). 

# PlantSeg from GUI
The graphical user interface is the easiest way to configure and run PlantSeg. 
Currently the GUI does not allow to visualize or interact with the data. 
We recommend using [MorphographX](https://www.mpipz.mpg.de/MorphoGraphX) or 
[Fiji](https://fiji.sc/) in order to assert the success and quality of the pipeline results.

### File Browser 
The file browser can be used to select the input file for the pipeline. 
PlantSeg can run on a single file (button A) or in batch mode for all files inside a directory (button B). 
If a directory is selected PlantSeg will run on all compatible files inside the directory. 

### Pipeline 
The central panel of PlantSeg (C) is the core of the pipeline configuration.
It can be used for customizing and tuning the pipeline accordingly to the data at hand. 
Detailed information for each stage can be found at:
* [Data-Processing](Data-Processing.md)
* [CNN-Predictions](Predictions.md)
* [Segmentation](Segmentation.md)

Any of the above widgets can be run singularly or in sequence (left to right). The order of execution can not be modified.


### Run 
The last panel has to main functions.
Running the pipeline (D), once the run button is pressed the
pipeline starts. The button is inactive until the process is finished.   
Adding a custom model (E). Custom trained model can be done by using the dedicated popup. Training a new model can be 
done following the instruction at [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet). 

### Start PlantSeg GUI
In Order to start the PlantSeg app in GUI mode:  
First, activate the newly created conda environment with:
```bash
conda activate plant-seg
```

then, run the GUI by simply typing:
```bash
$ plantseg --gui
```

# PlantSeg from configuration file
This modality of using PlantSeg is particularly suited for high throughput processing and for running
PlantSeg on a remote server. 
In order to use it one must create a configuration file using a normal text editor or using the save option of the
PlantSeg gui.

Here an example configuration:

```
path: /home/USERNAME/DATA.tiff # Contains the path to the directory or file to process

preprocessing:
  # enable/disable preprocessing
  state: True
  # create a new sub folder where all results will be stored
  save_directory: "PreProcessing"
  # rescaling the volume is essential for the generalization of the networks. The rescaling factor can be computed as the resolution
  # of the volume at hand divided by the resolution of the dataset used in training. Be careful, if the difference is too large check for a different model.
  factor: [1.0, 1.0, 1.0]
  # the order of the spline interpolation
  order: 2
  # optional: perform Gaussian smoothing or median filtering on the input.
  filter:
    # enable/disable filtering
    state: False
    # Accepted values: 'gaussian'/'median'
    type: gaussian
    # sigma (gaussian) or disc radius (median)
    param: 1.0

cnn_prediction:
  # enable/disable UNet prediction
  state: True
  # Trained model name, more info on available models and custom models in the README
  model_name: "generic_confocal_3d_unet"
  # If a CUDA capable gpu is available and corrected setup use "cuda", if not you can use "cpu" for cpu only inference (slower)
  device: "cpu"
  # (int or tuple) mirror pad the input stack in each axis for best prediction performance
  mirror_padding: [16, 32, 32]
  # how many subprocesses to use for data loading
  num_workers: 8
  # patch size given to the network (adapt to fit in your GPU mem)
  patch: [32, 128, 128]
  # stride between patches (make sure the the patches overlap in order to get smoother prediction maps)
  stride: [20, 100, 100]
  # "best" refers to best performing on the val set (recommended), alternatively "last" refers to the last version before interruption
  version: best
  # If "True" forces downloading networks from the online repos
  model_update: False

cnn_postprocessing:
  # enable/disable cnn post processing
  state: False
  # if True convert to result to tiff
  tiff: False
  # rescaling factor
  factor: [1, 1, 1]
  # spline order for rescaling
  order: 2

segmentation:
  # enable/disable segmentation
  state: True
  # Name of the algorithm to use for inferences. Options: MultiCut, MutexWS, GASP, DtWatershed
  name: "MultiCut"
  # Segmentation specific parameters here
  # balance under-/over-segmentation; 0 - aim for undersegmentation, 1 - aim for oversegmentation. (Not active for DtWatershed)
  beta: 0.5
  # directory where to save the results
  save_directory: "MultiCut"
  # enable/disable watershed
  run_ws: True
  # use 2D instead of 3D watershed
  ws_2D: True
  # probability maps threshold
  ws_threshold: 0.5
  # set the minimum superpixels size
  ws_minsize: 50
  # sigma for the gaussian smoothing of the distance transform
  ws_sigma: 2.0
  # sigma for the gaussian smoothing of boundary
  ws_w_sigma: 0
  # set the minimum segment size in the final segmentation. (Not active for DtWatershed)
  post_minsize: 50

segmentation_postprocessing:
  # enable/disable segmentation post processing
  state: False
  # if True convert to result to tiff
  tiff: False
  # rescaling factor
  factor: [1, 1, 1]
  # spline order for rescaling (keep 0 for segmentation post processing
  order: 0
```
This configuration can be found here [config.yaml](examples/config.yaml).

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
where `CONFIG_PATH` is the path to the YAML configuration file.