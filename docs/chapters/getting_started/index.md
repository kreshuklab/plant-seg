# Quick Start

PlantSeg has two main modes of operation: interactively, using the [Napari](https://napari.org/stable/index.html) viewer, or for batch processing from the command line.
The following sections will guide you through the installation and usage of
PlantSeg in each of these modes.

## Interactive PlantSeg with Napari Viewer

After [installing PlantSeg](installation.md) using the installer, there should
be a start menu entry to launch PlantSeg.
Alternatively, launch `plantseg --napari` in the terminal.

Using the GUI, you can

* [load and view images](../plantseg_interactive_napari/import.md)
* [apply pre- and post-processing steps and directly see the results](../plantseg_interactive_napari/preprocessing.md)
* [perform segmentation](../plantseg_interactive_napari/segmentation.md)
* [interactively proofread segmentations](../plantseg_interactive_napari/proofreading.md)
* [save the processing history as repeatable workflows](../plantseg_interactive_napari/batch.md)
* [train custom boundary prediction models](../plantseg_interactive_napari/unet_training.md)

Details to the GUI workflow and its sections can be found in the [GUI documentation](../plantseg_interactive_napari/index.md).  

## Run batch workflows

PlanSeg can apply a set of operations (a workflow) to a batch of images.  

To generate a batch workflow, you need to create a workflow `yaml` file.
While you process a single image, all steps are recorded.
Once you have saved the resulting image, you can save the recording as a workflow
file to be applied to many images.

Before you can run the batch workflow, you need to set the correct paths to the files
you want to process, and where you want the results to be saved.
PlantSeg provides an editor for this; start it from the `Output tab`, or by running:

```bash
plantseg --edit [workflow.yaml]
```

Once the correct input/output paths are set, you can run the workflow:

```bash
plantseg --config [workflow.yaml]
```

[Learn more about Workflows here!](../workflow_gui/index.md)
