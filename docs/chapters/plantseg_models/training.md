# Training on New Data

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
