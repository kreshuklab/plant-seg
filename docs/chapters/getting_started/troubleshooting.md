# Troubleshooting  <!-- omit in toc -->

- [`Could not load library libcudnn_ops_infer.so.8.`](#could-not-load-library-libcudnn_ops_inferso8)
- [`NVIDIA driver on your system is too old` or `Torch not compiled with CUDA enabled`](#nvidia-driver-on-your-system-is-too-old-or-torch-not-compiled-with-cuda-enabled)
- [`RuntimeError: key : KEY_NAME is missing, plant-seg requires KEY_NAME to run`](#runtimeerror-key--key_name-is-missing-plant-seg-requires-key_name-to-run)
- [`cannot import name 'lifted_problem_from_probabilities'`](#cannot-import-name-lifted_problem_from_probabilities)
- [Other issues](#other-issues)

----

#### `Could not load library libcudnn_ops_infer.so.8.`

If you stumble in the following error message:
```
Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
```

Just install `cudnn` by running:
```
$ mamba install -c conda-forge cudnn
```

----

#### `NVIDIA driver on your system is too old` or `Torch not compiled with CUDA enabled`

If you stumble in the following error message:
```
AssertionError:
The NVIDIA driver on your system is too old (found version xxxx).
Please update your GPU driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx
Alternatively, go to: http://pytorch.org to install
a PyTorch version that has been compiled with your version
of the CUDA driver.
```
or:
```
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```
It means that your cuda installation does not match the default in plantseg.
You can check your current cuda version by typing in the terminal
```
cat /usr/local/cuda/version.txt
```
Then you can re-install the pytorch version compatible with your cuda by activating your `plant-seg` environment:
```
conda activate plant-seg
```
and
```
conda install -c pytorch torchvision cudatoolkit=<YOU_CUDA_VERSION> pytorch
```
e.g. for cuda 9.2
```
conda install -c pytorch torchvision cudatoolkit=9.2 pytorch
```

Alternatively one can create the `plant-seg` environment from scratch and ensuring the correct version of cuda/pytorch, by:
```
conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c conda-forge cudatoolkit=<YOU_CUDA_VERSION> plantseg
```

----

#### `RuntimeError: key : KEY_NAME is missing, plant-seg requires KEY_NAME to run`

If you use plantseg from the GUI and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```
(or a similar message for any of the other keys)
It might be that the last session configuration file got corrupted or is outdated.
You should be able to solve it by removing the corrupted file `config_gui_last.yaml`.

If you have a standard installation of plantseg, you can remove it by executing on the terminal:
```
$ rm ~/.plantseg_models/configs/config_gui_last.yaml
```

If you use plantseg from the command line, and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```

Please make sure that your configuration has the correct formatting and contains all required keys.
An updated example can be found inside the directory `examples`, in this repository.

----

#### `cannot import name 'lifted_problem_from_probabilities'`

If when trying to execute the Lifted Multicut pipeline you receive an error like:
```
'cannot import name 'lifted_problem_from_probabilities' from 'elf.segmentation.features''
```
The solution is to re-install [elf](https://github.com/constantinpape/elf) via
```
conda install -c conda-forge python-elf
```

----

#### Other issues

* PlantSeg is under active development, so it may happen that the models/configuration files saved in `~/.plantseg_modes`
are outdated. In case of errors related to loading the configuration file, please close the PlantSeg app,
remove `~/.plantseg_models` directory and try again.
