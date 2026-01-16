import os

import yaml


def generate_script(checkpoint_dir, phase):
    assert phase in ["train", "test"]

    if phase == "train":
        script = "train.py"
    else:
        script = "predict.py"

    return f"""#!/bin/bash

#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 72:00:00                     
#SBATCH -o {checkpoint_dir}/{phase}.log			        
#SBATCH -e {checkpoint_dir}/error.log
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de
#SBATCH -p gpu				            
#SBATCH -C "gpu=2080Ti|gpu=1080Ti"		        
#SBATCH --gres=gpu:1	

module load cuDNN

export PYTHONPATH="/g/kreshuk/wolny/workspace/pytorch-3dunet:$PYTHONPATH"

/g/kreshuk/wolny/workspace/pytorch-3dunet/pytorch3dunet/{script} --config {checkpoint_dir}/config_{phase}.yml
"""


def _get_config_paths(root_dir):
    config_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            config_file = os.path.join(root, f)
            if config_file.endswith(".yml"):
                config_files.append(config_file)
    return config_files


if __name__ == "__main__":
    phase = "test"
    i = 1
    base_dir = "./grid_search"
    for config_file in _get_config_paths(base_dir):
        config_name = os.path.split(config_file)[1]
        if not (config_name == f"config_{phase}.yml"):
            continue
        print("Processing", config_file)
        config = yaml.safe_load(open(config_file, "r"))
        if phase == "train":
            checkpoint_dir = config["trainer"]["checkpoint_dir"]
        else:
            checkpoint_dir = os.path.split(config["model_path"])[0]
        slurm_script = generate_script(checkpoint_dir, phase)
        slurm_dir = os.path.join(base_dir, "slurm")
        os.makedirs(slurm_dir, exist_ok=True)
        with open(os.path.join(slurm_dir, f"{phase}_{i}.sh"), "w") as f:
            f.write(slurm_script)
        i += 1
