import os
import sys
import wget
import yaml
config_train = "config_train.yml"
best_model = "best_checkpoint.pytorch"
last_model = "last_checkpoint.pytorch"


def check_models(model_name, update_files=False):
    """
    Simple script to check and download trained modules
    """
    model_dir = os.path.join(os.path.expanduser("~"), ".plantseg_models", model_name)

    # Check if model directory exist if not create it
    if ~os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_config_path = os.path.exists(os.path.join(model_dir, config_train))
    model_best_path = os.path.exists(os.path.join(model_dir, best_model))
    model_last_path = os.path.exists(os.path.join(model_dir, last_model))

    # Check if files are there, if not download them
    if (not model_config_path or
            not model_best_path or
            not model_last_path or
            update_files):

        # Read config
        model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_zoo.yaml")
        config = yaml.load(open(model_file, 'r'), Loader=yaml.FullLoader)
        url = config[model_name]["path"]

        # wget models and training config
        temp_stdout = sys.stdout  # Hack! stdout has to go back to sys stdout because of progress bar
        sys.stdout = sys.__stdout__

        wget.download(url + config_train, out=model_dir)
        wget.download(url + best_model, out=model_dir)
        wget.download(url + last_model, out=model_dir)

        sys.stdout = temp_stdout  # return stdout to gui

    return 0
