import logging
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

from plantseg import (
    DIR_PLANTSEG_MODELS,
    FILE_CONFIG_TRAIN_YAML,
    PATH_HOME,
    PATH_TRAIN_TEMPLATE,
)
from plantseg.training.augs import Augmenter
from plantseg.training.h5dataset import HDF5Dataset
from plantseg.training.losses import DiceLoss
from plantseg.training.model import UNet2D, UNet3D
from plantseg.training.trainer import UNetTrainer

logger = logging.getLogger(__name__)


def create_model_config(
    checkpoint_dir: Path,
    in_channels,
    out_channels,
    patch_size,
    dimensionality,
    sparse,
    f_maps,
    max_num_iters,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(PATH_TRAIN_TEMPLATE, "r") as f:
        train_template = yaml.load(f, Loader=yaml.FullLoader)

    train_template["model"]["in_channels"] = in_channels
    train_template["model"]["out_channels"] = out_channels
    train_template["model"]["f_maps"] = f_maps
    if dimensionality == "2D":
        train_template["model"]["name"] = "UNet2D"
    else:
        train_template["model"]["name"] = "UNet3D"
    train_template["model"]["final_sigmoid"] = not sparse
    train_template["trainer"]["checkpoint_dir"] = str(checkpoint_dir)
    train_template["trainer"]["max_num_iterations"] = max_num_iters
    train_template["loaders"]["train"]["slice_builder"]["patch_shape"] = patch_size
    train_template["loaders"]["train"]["slice_builder"]["stride_shape"] = list(
        i // 2 for i in patch_size
    )
    train_template["loaders"]["val"]["slice_builder"]["patch_shape"] = patch_size
    train_template["loaders"]["val"]["slice_builder"]["stride_shape"] = patch_size

    out_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
    with open(out_path, "w") as yaml_file:
        yaml.dump(train_template, yaml_file, default_flow_style=False)


def unet_training(
    dataset_dir: str,
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: tuple,
    patch_size: tuple[int, int, int],
    max_num_iters: int,
    dimensionality: str,
    sparse: bool,
    device: str,
) -> None:
    # Model instantiation and logging
    final_sigmoid = not sparse
    if dimensionality in ["2D", "2d"]:
        model = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            f_maps=feature_maps,
            final_sigmoid=final_sigmoid,
        )
    else:
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            f_maps=feature_maps,
            final_sigmoid=final_sigmoid,
        )
    logger.info(f"Using {model.__class__.__name__} model for training.")

    # Device configuration
    batch_size = 1
    if torch.cuda.device_count() > 1 and device != "cpu":
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for prediction.")
        batch_size *= torch.cuda.device_count()
        device = "cuda"

    logger.info(f"Sending model to {device}")
    model = model.to(device)

    # Data loaders setup
    logger.info(f"Creating train/val loaders with batch size {batch_size}")
    train_datasets = create_datasets(dataset_dir, "train", patch_size)
    val_datasets = create_datasets(dataset_dir, "val", patch_size)
    loaders = {
        "train": DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        ),
        "val": DataLoader(
            ConcatDataset(val_datasets),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        ),
    }

    # Optimizer and training environment setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    checkpoint_dir = PATH_HOME / DIR_PLANTSEG_MODELS / model_name
    logger.info(f"Saving training files in {checkpoint_dir}")
    assert not checkpoint_dir.exists(), (
        f"Checkpoint dir {checkpoint_dir} already exists!"
    )

    create_model_config(
        checkpoint_dir,
        in_channels,
        out_channels,
        patch_size,
        dimensionality,
        sparse,
        feature_maps,
        max_num_iters,
    )

    # Trainer initialization and execution
    trainer = UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=ReduceLROnPlateau(optimizer, factor=0.2, patience=10),
        loss_criterion=DiceLoss(),
        loaders=loaders,
        checkpoint_dir=checkpoint_dir,
        max_num_iterations=max_num_iters,
        device=device,
    )

    trainer.train()


def create_datasets(dataset_dir: str, phase: str, patch_shape):
    assert phase in ["train", "val"], f"Phase {phase} not supported"
    phase_dir = Path(dataset_dir) / phase
    file_paths = find_h5_files(phase_dir)
    return [
        HDF5Dataset(file_path=file_path, augmenter=Augmenter(), patch_shape=patch_shape)
        for file_path in file_paths
    ]


def find_h5_files(data_dir):
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"Not a directory {data_dir}"
    return (
        list(data_dir.glob("*.h5"))
        + list(data_dir.glob("*.hdf"))
        + list(data_dir.glob("*.hdf5"))
        + list(data_dir.glob("*.hd5"))
    )
