import glob
import os
from itertools import chain
from typing import Tuple

import torch
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

from plantseg import PLANTSEG_MODELS_DIR
from plantseg.training.augs import Augmenter
from plantseg.training.h5dataset import HDF5Dataset
from plantseg.training.losses import DiceLoss
from plantseg.training.model import UNet2D, UNet3D
from plantseg.training.trainer import UNetTrainer


def unet_training(dataset_dir: str, model_name: str, in_channels: int, out_channels: int,
                  patch_size: Tuple[int, int, int], dimensionality: str,
                  sparse: bool, device: str, **kwargs) -> Image:
    # create loaders
    train_datasets = create_datasets(dataset_dir, 'train', patch_size)
    val_datasets = create_datasets(dataset_dir, 'val', patch_size)
    loaders = {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=True, pin_memory=True,
                            num_workers=4),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=1, shuffle=False, pin_memory=True,
                          num_workers=1)
    }

    # create model
    # set final activation to sigmoid if not sparse (i.e. not embedding model)
    final_sigmoid = not sparse
    if dimensionality == '2D':
        model = UNet2D(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid)
    else:
        model = UNet3D(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid)
    model = model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # create trainer
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=ReduceLROnPlateau(optimizer, factor=0.5, patience=10),
        loss_criterion=DiceLoss(),
        loaders=loaders,
        checkpoint_dir=os.path.join(PLANTSEG_MODELS_DIR, model_name),
        max_num_epochs=10000,
        max_num_iterations=100000,
        device=device
    )

    return trainer.train()


def create_datasets(dataset_dir, phase, patch_shape):
    assert phase in ['train', 'val'], f'Phase {phase} not supported'
    phase_dir = os.path.join(dataset_dir, phase)
    file_paths = traverse_h5_paths(phase_dir)
    return [HDF5Dataset(file_path=file_path, augmenter=Augmenter(), patch_shape=patch_shape) for file_path in
            file_paths]


def traverse_h5_paths(file_paths):
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results
