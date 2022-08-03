import numpy as np
import torch
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.utils import remove_halo
from torch.utils.data import DataLoader

from plantseg.predictions.array_dataset import ArrayDataset

logger = get_logger('UNetArrayPredictor')


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
    """

    def __init__(self, model, config, device, verbose_logging=True, **kwargs):
        self.model = model
        self.config = config
        self.device = device
        self.predictor_config = kwargs
        self.mute_logging = verbose_logging

    def __call__(self, test_dataset):
        assert isinstance(test_dataset, ArrayDataset)

        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.prediction_collate)

        if self.mute_logging:
            logger.info(f"Processing...")

        out_channels = self.config.get('out_channels')

        # prediction_channel = self.config.get('prediction_channel', None)
        prediction_channel = None
        if self.mute_logging and prediction_channel is not None:
            logger.info(f"Saving only channel '{prediction_channel}' from the network output")

        output_heads = self.config.get('output_heads', 1)

        if self.mute_logging:
            logger.info(f'Running prediction on {len(test_loader)} batches...')

        # dimensionality of the output predictions
        volume_shape = self.volume_shape(test_dataset)
        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        if self.mute_logging:
            logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (4, 8, 8))
        self._validate_halo(patch_halo, test_dataset.slice_builder_config)

        if self.mute_logging:
            logger.info(f'Using patch_halo: {patch_halo}')

        if self.mute_logging:
            # allocate prediction and normalization arrays
            logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads)

        # Sets the module in evaluation mode explicitly
        # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
        self.model.eval()
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in test_loader:
                # send batch to device
                batch = batch.to(self.device)

                # forward pass
                predictions = self.model(batch)

                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions,
                                                                          prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        index = (channel_slice,) + index

                        if self.mute_logging and prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        if self.mute_logging:
                            logger.info(f'Saving predictions for slice:{index}...')

                        # remove halo in order to avoid block artifacts in the output probability maps
                        u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results
        if self.mute_logging:
            logger.info(f'Returning predictions')
        prediction_maps = self._normalize_results(prediction_maps,
                                                  normalization_masks,
                                                  test_dataset.mirror_padding,
                                                  mute_logging=self.mute_logging)
        return prediction_maps

    @staticmethod
    def _allocate_prediction_maps(output_shape, output_heads):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    @staticmethod
    def _normalize_results(prediction_maps, normalization_masks, mirror_padding, mute_logging=False):
        def _slice_from_pad(pad):
            if pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        out_prediction_maps = []
        for prediction_map, normalization_mask in zip(prediction_maps, normalization_masks):
            prediction_map = prediction_map / normalization_mask

            if mirror_padding is not None:
                z_s, y_s, x_s = [_slice_from_pad(p) for p in mirror_padding]
                if mute_logging:
                    logger.info(f'Dataset loaded with mirror padding: {mirror_padding}. Cropping before saving...')

                prediction_map = prediction_map[:, z_s, y_s, x_s]

            out_prediction_maps.append(prediction_map)
        return out_prediction_maps

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(patch_overlap - patch_halo >= 0),\
            f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"

    @staticmethod
    def volume_shape(dataset):
        raw = dataset.raw
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]
