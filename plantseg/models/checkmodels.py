import os
import wget
import yaml
config_train = "/config_train.yml"
best_model = "/best_checkpoint.pytorch"
last_model = "/last_checkpoint.pytorch"


def check_models(model_name, update_files=False):
    """
    Simple script to check and download trained modules
    """
    model_dir = os.path.join(os.path.expanduser("~"), ".plantseg_models", model_name)

    # Check if model directory exist if not create it
    if ~os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Check if files are there, if not download them
    if (not os.path.exists(model_dir + config_train) or
            not os.path.exists(model_dir + best_model) or
            not os.path.exists(model_dir + last_model) or
            update_files):

        # Read config
        model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_zoo.yaml")
        config = yaml.load(open(model_file, 'r'), Loader=yaml.FullLoader)
        url = config[model_name]

        # wget models and training config
        wget.download(url + config_train[1:], out=model_dir)
        wget.download(url + best_model[1:], out=model_dir)
        wget.download(url + last_model[1:], out=model_dir)
    return 0
