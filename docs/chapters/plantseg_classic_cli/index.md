# PlantSeg Classic CLI

## Guide to Custom Configuration File

The configuration file defines all the operations in our pipeline together with the data to be processed.
Please refer to [config.yaml](https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml) for a sample pipeline configuration and a detailed explanation
of all parameters.

## Main Keys/Steps

* `path` attribute: is used to define either the file to process or the directory containing the data.
* `preprocessing` attribute: contains a simple set of possible operations one would need to run on their data before calling the neural network.
This step can be skipped if data is ready for neural network processing.
Detailed instructions can be found at [Classic GUI (Data Processing)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/data_processing/).
* `cnn_prediction` attribute: contains all parameters relevant for predicting with a neural network.
Description of all pre-trained models provided with the package is described below.
Detailed instructions can be found at [Classic GUI (Predictions)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/cnn_predictions/).
* `segmentation` attribute: contains all parameters needed to run the partitioning algorithm (i.e., final Segmentation).
Detailed instructions can be found at [Classic GUI (Segmentation)](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/segmentation/).

## Additional information

The PlantSeg-related files (models, configs) will be placed inside your home directory under `~/.plantseg_models`.

Our pipeline uses the PyTorch library for CNN predictions. PlantSeg can be run on systems without GPU, however
for maximum performance, we recommend that the application is run on a machine with a high-performance GPU for deep learning.
If the `CUDA_VISIBLE_DEVICES` environment variable is not specified, the prediction task will be distributed on all available GPUs.
E.g. run: `CUDA_VISIBLE_DEVICES=0 plantseg --config CONFIG_PATH` to restrict prediction to a given GPU.

## configuration file example

This modality of using PlantSeg is particularly suited for high throughput processing and for running
PlantSeg on a remote server.
To use PlantSeg from command line mode, you will need to create a configuration file using a standard text editor
 or using the save option of the PlantSeg GUI.

Here is an example configuration:

```yaml
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
  model_name: "generic_confocal_3D_unet"
  # If a CUDA capable gpu is available and corrected setup use "cuda", if not you can use "cpu" for cpu only inference (slower)
  device: "cpu"
  # how many subprocesses to use for data loading
  num_workers: 8
  # patch size given to the network (adapt to fit in your GPU mem)
  patch: [32, 128, 128]
  # stride between patches will be computed as `stride_ratio * patch`
  # recommended values are in range `[0.5, 0.75]` to make sure the patches have enough overlap to get smooth prediction maps
  stride_ratio: 0.75
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

This configuration can be found at [config.yaml](https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml).

## Pipeline Usage (command line)

To start PlantSeg from the command line. First, activate the newly created conda environment with:

```bash
conda activate plant-seg
```

then, one can just start the pipeline with

```bash
plantseg --config CONFIG_PATH
```

where `CONFIG_PATH` is the path to a YAML configuration file.

## Data Parallelism

In the headless mode (i.e. when invoked with `plantseg --config CONFIG_PATH`) the prediction step will run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.

```bash
CUDA_VISIBLE_DEVICES=0,1 plantseg --config CONFIG_PATH
```

## Results

The results are stored together with the source input files inside a nested directory structure.
As an example, if we want to run PlantSeg inside a directory with two stacks, we will obtain the following
outputs:

```bash
/file1.tif
/file2.tif
/PreProcesing/
------------>/file1.h5
------------>/file1.yaml
------------>/file2.h5
------------>/file2.yaml
------------>/generic_confocal_3d_unet/
------------------------------------->/file1_predictions.h5
------------------------------------->/file1_predictions.yaml
------------------------------------->/file2_predictions.h5
------------------------------------->/file2_predictions.yaml
------------------------------------->/GASP/
------------------------------------------>/file_1_predions_gasp_average.h5
------------------------------------------>/file_1_predions_gasp_average.yaml
------------------------------------------>/file_2_predions_gasp_average.h5
------------------------------------------>/file_2_predions_gasp_average.yaml
------------------------------------------>/PostProcessing/
--------------------------------------------------------->/file_1_predions_gasp_average.tiff
--------------------------------------------------------->/file_1_predions_gasp_average.yaml
--------------------------------------------------------->/file_2_predions_gasp_average.tiff
--------------------------------------------------------->/file_2_predions_gasp_average.yaml
```

The use of this hierarchical directory structure allows PlantSeg to find the necessary files quickly and can be used
to test different segmentation algorithms/parameter combinations minimizing the memory overhead on the disk.
For the sake of reproducibility, every file is associated with a configuration file ".yaml" that saves all parameters used
to produce the result.

## LiftedMulticut segmentation

As reported in our [paper](https://elifesciences.org/articles/57613), if one has a nuclei signal imaged together with
the boundary signal, we could leverage the fact that one cell contains only one nucleus and use the `LiftedMultict`
segmentation strategy and obtain improved segmentation.
We will use the _Arabidopsis thaliana_ lateral root as an example. The `LiftedMulticut` strategy consists of running
PlantSeg two times:

1. Using PlantSeg to predict the nuclei probability maps using the `lightsheet_unet_bce_dice_nuclei_ds1x` network.
In this case, only the pre-processing and CNN prediction steps are enabled in the config. See [example nuclei prediction config](https://github.com/kreshuklab/plant-seg/blob/master/plantseg/resources/nuclei_predictions_example.yaml).

    ```bash
    plantseg --config nuclei_predictions_example.yaml
    ```

1. Using PlantSeg to segment the input image with the `LiftedMulticut` algorithm given the nuclei probability maps from the 1st step.
See [example lifted multicut config](https://github.com/kreshuklab/plant-seg/blob/master/plantseg/resources/lifted_multicut_example.yaml). The notable difference is that in the `segmentation`
part of the config, we set `name: LiftedMulticut` and the `nuclei_predictions_path` as the path to the directory where the nuclei pmaps
were saved in step 1. Also, make sure that the `path` attribute points to the raw files containing the cell boundary staining (NOT THE NUCLEI).

    ```bash
    plantseg --config lifted_multicut_example.yaml
    ```

If case when the nuclei segmentation is given, one should skip step 1., add `is_segmentation=True` flag in the [example lifted multicut config](https://github.com/kreshuklab/plant-seg/blob/master/plantseg/resources/lifted_multicut_example.yaml)
and directly run step 2.
