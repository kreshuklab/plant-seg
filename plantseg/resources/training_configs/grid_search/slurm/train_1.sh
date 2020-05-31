#!/bin/bash

#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 72:00:00                     
#SBATCH -o /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/ovules/resunet_gn_bce/train.log			        
#SBATCH -e /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/ovules/resunet_gn_bce/error.log
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de
#SBATCH -p gpu				            
#SBATCH -C "gpu=2080Ti|gpu=1080Ti"		        
#SBATCH --gres=gpu:1	

module load cuDNN

export PYTHONPATH="/g/kreshuk/wolny/workspace/pytorch-3dunet:$PYTHONPATH"

/g/kreshuk/wolny/workspace/pytorch-3dunet/pytorch3dunet/train.py --config /g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/./grid_search/ovules/resunet_gn_bce/config_train.yml
