from typing import Tuple

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from plantseg.training.embeddings import embeddings_to_affinities
from plantseg.training.model import UNet2D
from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.array_dataset import ArrayDataset, default_prediction_collate


def _is_2d_model(model: nn.Module) -> bool:
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


def _pad(m: torch.Tensor, patch_halo: Tuple[int, int, int]) -> torch.Tensor:
    if patch_halo is not None:
        z, y, x = patch_halo
        return nn.functional.pad(m, (x, x, y, y, z, z), mode='reflect')
    return m


def _unpad(m: torch.Tensor, patch_halo: Tuple[int, int, int]) -> torch.Tensor:
    if patch_halo is not None:
        z, y, x = patch_halo
        if z == 0:
            return m[..., y:-y, x:-x]
        else:
            return m[..., z:-z, y:-y, x:-x]
    return m


def find_batch_size(model: nn.Module, in_channels: int, patch_shape: Tuple[int, int, int],
                    patch_halo: Tuple[int, int, int], device: str) -> int:
    if device == 'cpu':
        return 1

    if isinstance(model, UNet2D):
        patch_shape = patch_shape[1:]

    patch_shape = tuple(patch_shape)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_size in [2, 4, 8, 16, 32, 64, 128]:
            try:
                x = torch.randn((batch_size, in_channels) + patch_shape).to(device)
                x = _pad(x, patch_halo)
                _ = model(x)
            except RuntimeError as e:
                batch_size //= 2
                break

        del model
        torch.cuda.empty_cache()
        return batch_size


class ArrayPredictor:
    """
    Based on pytorch-3dunet StandardPredictor
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py

    Applies the model on the given dataset and returns the results as a list of numpy arrays.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.
    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        in_channels (int): number of input channels to the model
        out_channels (int): number of output channels from the model
        device (str): device to use for prediction
        patch (tuple): patch size to use for prediction
        patch_halo (tuple): mirror padding around the patch
        single_batch_mode (bool): if True, the batch size will be set to 1
        headless (bool): if True, DataParallel will be used if multiple GPUs are available
        is_embedding (bool): if True, the model returns embeddings instead of probabilities
    """

    def __init__(self, model: nn.Module, in_channels: int, out_channels: int, device: str, patch: Tuple[int, int, int],
                 patch_halo: Tuple[int, int, int], single_batch_mode: bool, headless: bool, is_embedding: bool = False,
                 verbose_logging: bool = False, disable_tqdm: bool = False):

        self.device = device
        if single_batch_mode:
            self.batch_size = 1
        else:
            self.batch_size = find_batch_size(model, in_channels, patch, patch_halo, device)
        gui_logger.info(f'Using batch size of {self.batch_size} for prediction')

        if torch.cuda.device_count() > 1 and device != 'cpu' and headless:
            model = nn.DataParallel(model)
            gui_logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction. '
                            f'Increasing batch size to {torch.cuda.device_count()} * {self.batch_size}')
            self.batch_size *= torch.cuda.device_count()
            self.device = 'cuda'

        self.model = model.to(self.device)
        self.out_channels = out_channels
        self.patch_halo = patch_halo
        self.verbose_logging = verbose_logging
        self.disable_tqdm = disable_tqdm
        self.is_embedding = is_embedding

    def __call__(self, test_dataset: Dataset) -> np.ndarray:
        assert isinstance(test_dataset, ArrayDataset)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True,
                                 collate_fn=default_prediction_collate)

        if self.verbose_logging:
            gui_logger.info(f'Running prediction on {len(test_loader)} batches')

        # dimensionality of the output predictions
        volume_shape = self.volume_shape(test_dataset)
        is_2d_model = _is_2d_model(self.model)
        if self.is_embedding:
            if is_2d_model:
                # outputs 1-affinities in XY
                out_channels = 2
            else:
                # outputs 1-affinities in XYZ
                out_channels = 3
        else:
            out_channels = self.out_channels

        prediction_maps_shape = (out_channels,) + volume_shape

        if self.verbose_logging:
            gui_logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')
            gui_logger.info(f'Using patch_halo: {self.patch_halo}')
            # allocate prediction and normalization arrays
            gui_logger.info('Allocating prediction and normalization arrays...')

        # initialize the output prediction arrays
        prediction_map = np.zeros(prediction_maps_shape, dtype='float32')
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_mask = np.zeros(prediction_maps_shape, dtype='uint8')

        # run prediction
        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run predictions on the entire input dataset

        with torch.no_grad():
            for input, indices in tqdm.tqdm(test_loader, disable=self.disable_tqdm):
                input = input.to(self.device)
                # pad input patch
                input = _pad(input, self.patch_halo)
                # forward pass
                if is_2d_model:
                    # remove the singleton z-dimension from the input
                    input = torch.squeeze(input, dim=-3)
                    prediction = self.model(input)
                    # add the singleton z-dimension to the output
                    prediction = torch.unsqueeze(prediction, dim=-3)
                else:
                    prediction = self.model(input)

                if self.is_embedding:
                    if is_2d_model:
                        offsets = [[-1, 0], [0, -1]]
                    else:
                        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                    # convert embeddings to affinities
                    prediction = embeddings_to_affinities(prediction, offsets, delta=0.5)
                    # average across channels and invert (i.e. 1-affinities)
                    prediction = 1 - prediction.mean(dim=1)
                # unpad the prediction
                prediction = _unpad(prediction, self.patch_halo)
                # convert to numpy array
                prediction = prediction.cpu().numpy()
                channel_slice = slice(0, out_channels)
                # for each batch sample
                for pred, index in zip(prediction, indices):
                    # add channel dimension to the index
                    index = (channel_slice,) + tuple(index)
                    # accumulate probabilities into the output prediction array
                    prediction_map[index] += pred
                    # count voxel visits for normalization
                    normalization_mask[index] += 1

        if self.verbose_logging:
            gui_logger.info(f'Prediction finished')

        # normalize results and return
        return prediction_map / normalization_mask

    @staticmethod
    def volume_shape(dataset):
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]
