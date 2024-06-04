# Training on New Data

For training new models we rely on the [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).
A similar configuration file can be used for training on new data and all the instructions can be found in the repo.

## Adding Models

1. Put these three files in one directory:
      1. configuration file used for training: `config_train.yml`
      2. snapshot of the best model across training: `best_checkpoint.pytorch`
      3. snapshot of the last model saved during training: `last_checkpoint.pytorch`
2. Click "Add Custom Model" in the GUI and follow the instruction

### Alternative Old Method

When the network is trained, it is enough to create `~/.plantseg_models/MY_MODEL_NAME` directory
and copy the following files into it:

* configuration file used for training: `config_train.yml`
* snapshot of the best model across training: `best_checkpoint.pytorch`
* snapshot of the last model saved during training: `last_checkpoint.pytorch`

The later two files are automatically generated during training and contain all neural networks parameters.

Now you can simply use your model for prediction by setting the [config.yaml](https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml) key to `MY_MODEL_NAME`.

If you want your model to be part of the open-source model zoo provided with this package, please contact us.
