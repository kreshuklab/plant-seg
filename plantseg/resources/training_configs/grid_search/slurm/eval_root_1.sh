#!/bin/bash

#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 12:00:00                     
#SBATCH -o /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/root/resunet_gn_bce/predictions/eval.log			        
#SBATCH -e /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/root/resunet_gn_bce/predictions/error.log
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de	

export PYTHONPATH="/g/kreshuk/wolny/workspace/plant-seg:$PYTHONPATH"

/g/kreshuk/wolny/workspace/plant-seg/evaluation/evaluation_pmaps.py --gt /g/kreshuk/wolny/Datasets/LateralRoot/Test --predictions /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/root/resunet_gn_bce/predictions --threshold 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --sigma 1.0 --out-file /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/root/resunet_gn_bce
