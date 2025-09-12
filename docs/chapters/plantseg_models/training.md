# Advanced training

!!! warning
    The recommended way to train models is through the [napari training GUI](../plantseg_interactive_napari/unet_training.md).
    This section gives additional background information not needed for intended training procedure.

For training models, we rely on the [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).  
You can import models trained using this library into PlantSeg by choosing
`ADD CUSTOM MODEL` in the model selection drop-down in the segmentation tab.
Models are by default stored in `~/.plantseg_models/`.

Each model needs the following files:

* Configuration file used for training: `config_train.yml`
* A snapshot of the best model across training: `best_checkpoint.pytorch`
* A Snapshot of the last model saved during training: `last_checkpoint.pytorch`

Check the model directory to see examples of the `config_train.yml`
