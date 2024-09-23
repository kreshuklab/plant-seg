# PlantSeg from GUI

!!! failure "Deprecated"
    This interface is deprecated and has been removed from PlantSeg v2. Please use the Napari viewer or the command line interface instead, or install PlantSeg v1.

The graphical user interface is the easiest way to configure and run PlantSeg.
Currently the GUI does not allow to visualize or interact with the data.
We recommend using [MorphographX](https://www.mpipz.mpg.de/MorphoGraphX) or
[Fiji](https://fiji.sc/) in order to assert the success and quality of the pipeline results.

![alt text](https://github.com/kreshuklab/plant-seg/raw/assets/images/plantseg_overview.png)

## File Browser Widget

The file browser can be used to select the input files for the pipeline.
PlantSeg can run on a single file (button A) or in batch mode for all files inside a directory (button B).
If a directory is selected PlantSeg will run on all compatible files inside the directory.

## Main Pipeline Configurator

The central panel of PlantSeg (C) is the core of the pipeline configuration.
It can be used for customizing and tuning the pipeline accordingly to the data at hand.
Detailed information for each stage can be found at:

* [Data-Processing](data_processing.md)
* [CNN-Prediction](cnn_prediction.md)
* [Segmentation](segmentation.md)

Any of the above widgets can be run singularly or in sequence (left to right). The order of execution can not be
modified.

## Run

The last panel has two main functions.
Running the pipeline (D), once the run button is pressed the
pipeline starts. The button is inactive until the process is finished.
Adding a custom model (E). Custom trained model can be done by using the dedicated popup. Training a new model can be
done following the instruction at [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).

## Results

The results are stored together with the source input files inside a nested directory structure.
As example, if we want to run PlantSeg inside a directory with 2 stacks, we will obtain the following
outputs:

```
/file1.tif
/file2.tif
/PreProcesing/
------------>/file1.h5
------------>/file1.yaml
------------>/file2.h5
------------>/file2.yaml
------------>/generic_confocal_3d_unet/
------------------------------------->/file1_prediction.h5
------------------------------------->/file1_prediction.yaml
------------------------------------->/file2_prediction.h5
------------------------------------->/file2_prediction.yaml
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

The use of this hierarchical directory structure allows PlantSeg to easily find the necessary files and can be used
to test different combination of segmentation algorithms/parameters minimizing the memory overhead on the disk.
For sake of reproducibility, every file is associated with a configuration file ".yaml" that saves all parameters used
to produce the result.

## Start PlantSeg GUI

In order to start the PlantSeg app in GUI mode:
First, activate the newly created conda environment with:

```bash
conda activate plant-seg
```

then, run the GUI by simply typing:

```bash
plantseg --gui
```
