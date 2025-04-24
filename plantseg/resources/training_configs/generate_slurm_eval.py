import os

BASE_DIR = "/g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/"

GT_PATHS = {
    "root": "/g/kreshuk/wolny/Datasets/LateralRoot/Test",
    "ovules": "/g/kreshuk/wolny/Datasets/Ovules/test/Lorenzo/ds2",
}


def generate_script(net_path, dataset):
    predictions_path = os.path.join(net_path, "predictions")
    return f"""#!/bin/bash

#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 12:00:00                     
#SBATCH -o {predictions_path}/eval.log			        
#SBATCH -e {predictions_path}/error.log
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de	

export PYTHONPATH="/g/kreshuk/wolny/workspace/plant-seg:$PYTHONPATH"

/g/kreshuk/wolny/workspace/plant-seg/evaluation/evaluation_pmaps.py --gt {GT_PATHS[dataset]} --predictions {predictions_path} --threshold 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --sigma 1.0 --out-file {net_path}
"""


def _get_net_paths(base_dir, dataset):
    results = []
    ds_base = os.path.join(base_dir, dataset)
    for net_name in os.listdir(ds_base):
        net_path = os.path.join(BASE_DIR, ds_base, net_name)
        results.append(net_path)
    return results


if __name__ == "__main__":
    dataset = "root"
    i = 1
    base_dir = "./grid_search"
    for net_path in _get_net_paths(base_dir, dataset):
        slurm_script = generate_script(net_path, dataset)
        slurm_dir = os.path.join(base_dir, "slurm")
        os.makedirs(slurm_dir, exist_ok=True)
        with open(os.path.join(slurm_dir, f"eval_{dataset}_{i}.sh"), "w") as f:
            f.write(slurm_script)
        i += 1
