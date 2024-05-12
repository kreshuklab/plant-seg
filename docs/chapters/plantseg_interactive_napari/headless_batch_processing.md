# Headless Batch Processing

When image(s) are exported from PlantSeg Napari, a workflow file, e.g. `workflow.pkl`, describing the processing steps is saved alongside the exported images. This workflow can be used to process more images with the same parameters in each step of the workflow, in the headless mode.

To run the headless mode, use:

```bash
plantseg --headless PATH_TO_WORKFLOW
```

!!! warning "Network Prediction with 2-channel Output"
    If the network used in the workflow has a 2-channel output, no other steps can be performed after the network prediction step. The only supported PlantSeg Napari workflow for headless 2-channel-output prediction is open file -> network prediction -> save file.
