import os

import h5py
import numpy as np

from plantseg.pipeline.raw2seg import configure_cnn_step


class TestUnetPredictions:
    def test_state_false(self, prediction_config):
        prediction_config['cnn_prediction']['state'] = False
        paths = [prediction_config['path']]
        step = configure_cnn_step(paths, prediction_config['cnn_prediction'])

        assert paths == step()

    def test_unet_predictions(self, prediction_config):
        """
        Test Unet predictions including: network download, predictions on real data, and validating the output saved correctly
        """
        paths = [prediction_config['path']]
        cnn_config = prediction_config['cnn_prediction']
        step = configure_cnn_step(paths, prediction_config['cnn_prediction'])

        output_paths = step()

        # assert output_paths correctly created
        basepath, basename = os.path.split(paths[0])
        expected_paths = [
            os.path.join(basepath, cnn_config['model_name'], os.path.splitext(basename)[0] + '_predictions.h5')
        ]
        assert expected_paths == output_paths

        # assert predictions dataset exist in the output h5 and has correct voxel size
        with h5py.File(paths[0], 'r') as f:
            expected_voxel_size = f['raw'].attrs['element_size_um']

        with h5py.File(output_paths[0], 'r') as f:
            assert 'predictions' in f
            assert np.allclose(expected_voxel_size, f['predictions'].attrs['element_size_um'])
