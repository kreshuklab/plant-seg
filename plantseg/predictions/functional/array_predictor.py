import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from plantseg.models.model import UNet2D
from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.array_dataset import ArrayDataset, default_prediction_collate


def _is_2d_model(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


def _pad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        return nn.functional.pad(m, (x, x, y, y, z, z), mode='reflect')
    return m


def _unpad(m, patch_halo):
    if patch_halo is not None:
        z, y, x = patch_halo
        if z == 0:
            return m[..., y:-y, x:-x]
        else:
            return m[..., z:-z, y:-y, x:-x]
    return m


def get_batch_size(model):
    # TODO: implement
    return 1


class ArrayPredictor:
    """
    Based on pytorch-3dunet StandardPredictor
    https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/predictor.py

    Applies the model on the given dataset and returns the results as a list of numpy arrays.
    Predictions from the network are kept in memory. If the results from the network don't fit in into RAM
    use `LazyPredictor` instead.
    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        config (dict): global config dict
        device (str): device to use for prediction
        patch_halo (tuple): mirror padding around the patch
    """

    def __init__(self, model, config, device, patch_halo, verbose_logging=False, disable_tqdm=False):
        self.model = model
        self.config = config
        self.device = device
        self.patch_halo = patch_halo
        self.verbose_logging = verbose_logging
        self.disable_tqdm = disable_tqdm

    def __call__(self, test_dataset):
        assert isinstance(test_dataset, ArrayDataset)
        batch_size = get_batch_size(self.model)
        if torch.cuda.device_count() > 1 and self.device != 'cpu':
            gui_logger.info(f'{torch.cuda.device_count()} GPUs available. '
                            f'Using batch_size = {torch.cuda.device_count()} * {batch_size}')
            batch_size = batch_size * torch.cuda.device_count()

        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                 collate_fn=default_prediction_collate)

        if self.verbose_logging:
            gui_logger.info(f'Running prediction on {len(test_loader)} batches')

        # dimensionality of the output predictions
        volume_shape = self.volume_shape(test_dataset)
        out_channels = self.config['out_channels']
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
                # send batch to gpu
                if torch.cuda.is_available() and self.device != 'cpu':
                    input = input.cuda(non_blocking=True)

                input = _pad(input, self.patch_halo)

                if _is_2d_model(self.model):
                    # remove the singleton z-dimension from the input
                    input = torch.squeeze(input, dim=-3)
                    # forward pass
                    prediction = self.model(input)
                    # add the singleton z-dimension to the output
                    prediction = torch.unsqueeze(prediction, dim=-3)
                else:
                    # forward pass
                    prediction = self.model(input)

                # unpad
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


def remove_halo(patch, index, shape, patch_halo):
    """
    Copied from: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/utils.py

    Remove `patch_halo` voxels around the edges of a given patch.

    Args:
        patch (numpy.ndarray): the patch to remove the halo from
        index (tuple): the position of the patch in the original image of shape `shape`
        shape (tuple): the shape of the original image
        patch_halo (tuple): the halo size in each dimension
    """
    assert len(patch_halo) == 3

    def _new_slices(slicing, max_size, pad):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad
            i_start = slicing.start + pad

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad if pad != 0 else 1
            i_stop = slicing.stop - pad

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, patch.shape[0])

    p_z, i_z = _new_slices(i_z, D, patch_halo[0])
    p_y, i_y = _new_slices(i_y, H, patch_halo[1])
    p_x, i_x = _new_slices(i_x, W, patch_halo[2])

    patch_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return patch[patch_index], index
