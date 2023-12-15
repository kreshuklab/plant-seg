# Troubleshooting

* If you stumble in the following error message:
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
conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c awolny -c conda-forge cudatoolkit=<YOU_CUDA_VERSION> plantseg
```

* If you use plantseg from the GUI and you receive an error similar to:
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

* If you use plantseg from the comand line and you receive an error similar to:
```
RuntimeError: key : 'crop_volume' is missing, plant-seg requires 'crop_volume' to run
```

Please make sure that your configuratiuon has the correct formatting and contains all required keys.
An updated example can be found inside the directory `examples`, in this repository.

* If when trying to execute the Lifted Multicut pipeline you receive an error like:
```
'cannot import name 'lifted_problem_from_probabilities' from 'elf.segmentation.features''
```
The solution is to re-install [elf](https://github.com/constantinpape/elf) via
```
conda install -c conda-forge python-elf
```

* PlantSeg is under active development so it may happen that the models/configuration files saved in `~/.plantseg_modes`
are outdated. In case of errors related to loading the configuration file, please close the PlantSeg app,
remove `~/.plantseg_models` directory and try again.