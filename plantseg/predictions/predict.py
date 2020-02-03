import importlib
import os
import sys
import time

gui_out = sys.stdout  # Hack! stdout has to go back to sys stdout because of progress bar
sys.stdout = sys.__stdout__
from pytorch3dunet.datasets.hdf5 import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
sys.stdout = gui_out


def _get_output_file(dataset, model_name, suffix='_predictions'):
    basepath, basename = os.path.split(dataset.file_path)
    basename = f"{os.path.splitext(basename)[0]}{suffix}.h5"
    os.makedirs(os.path.join(basepath, model_name), exist_ok=True)
    return os.path.join(basepath, model_name, basename)


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)


class ModelPredictions:
    def __init__(self, config):

        self.gui_out = sys.stdout  # Hack! stdout has to go back to sys stdout because of progress bar
        sys.stdout = sys.__stdout__

        self.logger = utils.get_logger('UNet3DPredictor')

        self.config = config

        # Create the model
        model = get_model(config)
        self.path_out = []

        # Load model state
        model_path = config['model_path']
        self.model_name = config["model_name"]

        self.logger.info(f'Loading model from {model_path}...')
        utils.load_checkpoint(model_path, model)
        self.logger.info(f"Sending the model to '{config['device']}'")
        self.model = model.to(config['device'])

        self.logger.info('Loading HDF5 datasets...')

    def __call__(self):
        for test_loader in get_test_loaders(self.config):
            print(f"Predicting {test_loader.dataset.file_path}", file=self.gui_out)
            runtime = time.time()

            self.logger.info(f"Processing '{test_loader.dataset.file_path}'...")
            output_file = _get_output_file(test_loader.dataset, self.model_name)
            predictor = _get_predictor(self.model, test_loader, output_file, self.config)
            # run the model prediction on the entire dataset and save to the 'output_file' H5
            predictor.predict()
            self.path_out.append(output_file)

            runtime = time.time() - runtime
            print(" - Predicting took {} s".format(runtime), file=self.gui_out)

        sys.stdout = self.gui_out
        return self.path_out
