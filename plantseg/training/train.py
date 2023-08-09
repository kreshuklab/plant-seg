import glob
import os
from itertools import chain
from typing import Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from plantseg import PLANTSEG_MODELS_DIR, PLANTSEG_GLOBAL_PATH
from plantseg.pipeline import gui_logger
from plantseg.training.augs import Augmenter
from plantseg.training.h5dataset import HDF5Dataset
from plantseg.training.losses import DiceLoss
from plantseg.training.model import UNet2D, UNet3D
from plantseg.training.trainer import UNetTrainer


def create_model_config(checkpoint_dir, in_channels, out_channels, patch_size, dimensionality, sparse, f_maps):
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_template_path = os.path.join(PLANTSEG_GLOBAL_PATH,
                                       "resources",
                                       "config_train_template.yaml")
    with open(train_template_path, 'r') as f:
        train_template = yaml.load(f, Loader=yaml.FullLoader)

    train_template['model']['in_channels'] = in_channels
    train_template['model']['out_channels'] = out_channels
    train_template['model']['f_maps'] = f_maps
    if dimensionality == '2D':
        train_template['model']['name'] = 'UNet2D'
    else:
        train_template['model']['name'] = 'UNet3D'
    train_template['model']['final_sigmoid'] = not sparse
    train_template['trainer']['checkpoint_dir'] = checkpoint_dir
    train_template['loaders']['train']['slice_builder']['patch_shape'] = patch_size
    train_template['loaders']['val']['slice_builder']['patch_shape'] = patch_size

    out_path = os.path.join(checkpoint_dir, 'config_train.yml')
    with open(out_path, 'w') as yaml_file:
        yaml.dump(train_template, yaml_file, default_flow_style=False)


def plot_curves(learning_curves, checkpoint_dir):
    plt.figure()
    plt.plot(list(learning_curves['train_loss'].keys()), list(learning_curves['train_loss'].values()),
             label='train_loss', c='y', marker='o')
    plt.plot(list(learning_curves['val_loss'].keys()), list(learning_curves['val_loss'].values()),
             label='val_loss', c='b', marker='o')
    plt.legend(loc='upper right')
    plot_path = os.path.join(checkpoint_dir, 'learning_curves.png')
    plt.savefig(plot_path)
    return np.array(Image.open(plot_path))


def unet_training(dataset_dir: str, model_name: str, in_channels: int, out_channels: int,
                  patch_size: Tuple[int, int, int], max_num_iters: int, dimensionality: str,
                  sparse: bool, device: str, headless: bool = False, **kwargs) -> Image:
    # create model
    batch_size = 1
    # set final activation to sigmoid if not sparse (i.e. not embedding model)
    final_sigmoid = not sparse
    f_maps = [32, 64, 128, 256, 512]
    if dimensionality == '2D':
        model = UNet2D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, final_sigmoid=final_sigmoid)
    else:
        model = UNet3D(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, final_sigmoid=final_sigmoid)

    if torch.cuda.device_count() > 1 and device != 'cpu' and headless:
        model = nn.DataParallel(model)
        gui_logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction.')
        batch_size *= torch.cuda.device_count()
        device = 'cuda'

    model = model.to(device)
    # create loaders
    train_datasets = create_datasets(dataset_dir, 'train', patch_size)
    val_datasets = create_datasets(dataset_dir, 'val', patch_size)
    loaders = {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=4),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=1)
    }

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # create trainer
    home_path = os.path.expanduser("~")
    checkpoint_dir = os.path.join(home_path, PLANTSEG_MODELS_DIR, model_name)
    if os.path.exists(checkpoint_dir):
        gui_logger.warn(f'Checkpoint dir {checkpoint_dir} already exists! Overwriting...')
    create_model_config(checkpoint_dir, in_channels, out_channels, patch_size, dimensionality, sparse, f_maps)

    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=ReduceLROnPlateau(optimizer, factor=0.2, patience=10),
        loss_criterion=DiceLoss(),
        loaders=loaders,
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=max_num_iters,
        device=device
    )

    learning_curves = trainer.train()
    return plot_curves(learning_curves, checkpoint_dir)


def create_datasets(dataset_dir, phase, patch_shape):
    assert phase in ['train', 'val'], f'Phase {phase} not supported'
    phase_dir = os.path.join(dataset_dir, phase)
    file_paths = find_h5_files(phase_dir)
    return [HDF5Dataset(file_path=file_path, augmenter=Augmenter(), patch_shape=patch_shape) for file_path in
            file_paths]


def find_h5_files(data_dir):
    assert os.path.isdir(data_dir), f'Not a directory {data_dir}'
    results = []
    iters = [glob.glob(os.path.join(data_dir, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
    for fp in chain(*iters):
        results.append(fp)
    return results
