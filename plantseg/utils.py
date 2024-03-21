import os
import shutil
from pathlib import Path
from warnings import warn

import requests
import yaml

from plantseg import PATH_HOME, DIR_PLANTSEG_MODELS, PATH_PLANTSEG_GLOBAL
from plantseg.__version__ import __version__ as current_version
from plantseg.pipeline import gui_logger

CONFIG_TRAIN_YAML = "config_train.yml"
BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"


def load_config(config_path: str | Path) -> dict:
    """load a yaml config in a dictionary"""
    config_path = Path(config_path)
    config_content = config_path.read_text()
    config = yaml.load(config_content, Loader=yaml.FullLoader)
    return config


def save_config(config: dict, config_path: str | Path) -> None:
    """save a dictionary in a yaml file"""
    config_path = Path(config_path)
    with config_path.open('w') as f:
        yaml.dump(config, f)


def get_train_config(model_name: str) -> dict:
    """
    Load the training configuration of a model in the model zoo

    Args:
        model_name: name of the model in the model zoo

    Returns:
        the training config
    """
    check_models(model_name, config_only=True)
    # Load train config and add missing info
    train_config_path = os.path.join(PATH_HOME,
                                     DIR_PLANTSEG_MODELS,
                                     model_name,
                                     CONFIG_TRAIN_YAML)

    config_train = load_config(train_config_path)
    return config_train


def download_model_files(model_url: str, out_dir: str) -> None:
    model_file = model_url.split('/')[-1]
    config_url = model_url[:-len(model_file)] + "config_train.yml"
    urls = {
        "best_checkpoint.pytorch": model_url,
        "config_train.yml": config_url
    }
    download_files(urls, out_dir)


def download_model_config(model_url: str, out_dir: str) -> None:
    model_file = model_url.split('/')[-1]
    config_url = model_url[:-len(model_file)] + "config_train.yml"
    urls = {
        "config_train.yml": config_url
    }
    download_files(urls, out_dir)


def download_files(urls: dict, out_dir: str) -> None:
    for filename, url in urls.items():
        with requests.get(url, allow_redirects=True) as r:
            with open(os.path.join(out_dir, filename), 'wb') as f:
                f.write(r.content)


def check_models(model_name: str, update_files: bool = False, config_only: bool = False) -> bool:
    """
    Simple script to check and download trained modules
    :param model_name: name of the model in the model zoo
    :param update_files: if true force the re-download of the model
    :param config_only: if true only downloads the config file and skips the model file
    """

    if os.path.isdir(model_name):
        model_dir = model_name
    else:
        model_dir = os.path.join(os.path.expanduser("~"), DIR_PLANTSEG_MODELS, model_name)
        # Check if model directory exist if not create it
        if ~os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    model_config_path = os.path.exists(os.path.join(model_dir, CONFIG_TRAIN_YAML))
    model_best_path = os.path.exists(os.path.join(model_dir, BEST_MODEL_PYTORCH))

    # Check if files are there, if not download them
    if (not model_config_path or
            not model_best_path or
            update_files):

        # Read config
        model_file = os.path.join(PATH_PLANTSEG_GLOBAL, "resources", "models_zoo.yaml")
        config = load_config(model_file)

        if model_name in config:
            model_url = config[model_name]["model_url"]
            if config_only:
                gui_logger.info(f"Downloading model config...")
                download_model_config(model_url, out_dir=model_dir)
            else:
                gui_logger.info(f"Downloading model files: '{model_url}' ...")
                download_model_files(model_url, out_dir=model_dir)
        else:
            raise RuntimeError(f"Custom model {model_name} corrupted. Required files not found.")
    return True


def clean_models():
    for _ in range(3):
        answer = input("This will delete all models in the model zoo, "
                       "make sure to copy all custom models you want to preserve before continuing.\n"
                       "Are you sure you want to continue? (y/n) ")
        if answer == 'y':
            ps_models_dir = os.path.join(PATH_HOME, DIR_PLANTSEG_MODELS)
            shutil.rmtree(ps_models_dir)
            print("All models deleted... PlantSeg will now close")
            return None

        elif answer == 'n':
            print("Nothing was deleted.")
            return None

        else:
            print("Invalid input, please type 'y' or 'n'.")


def check_version(plantseg_url='https://api.github.com/repos/hci-unihd/plant-seg/releases/latest'):
    try:
        response = requests.get(plantseg_url).json()
        latest_version = response['tag_name']

    except requests.exceptions.ConnectionError:
        warn("Connection error, could not check for new version.")
        return None
    except requests.exceptions.Timeout:
        warn("Connection timeout, could not check for new version.")
        return None
    except requests.exceptions.TooManyRedirects:
        warn("Too many redirects, could not check for new version.")
        return None
    except Exception as e:
        warn(f"Unknown error, could not check for new version. Error: {e}")
        return None

    latest_version_numeric = [int(x) for x in latest_version.split(".")]
    plantseg_version_numeric = [int(x) for x in current_version.split(".")]

    if len(latest_version_numeric) != len(plantseg_version_numeric):
        warn(f"Could not check for new version, version number not in the correct format.\n"
             f"Current version: {current_version}, latest version: {latest_version}")
        return None

    for l_v, p_v in zip(latest_version_numeric, plantseg_version_numeric):
        if l_v > p_v:
            print(f"New version of PlantSeg available: {latest_version}.\n"
                  f"Please update your version to the latest one!")
            return None
