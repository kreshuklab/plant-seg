import glob
import os
import shutil
from pathlib import Path
from shutil import copy2
from typing import Tuple, Optional
from warnings import warn

import requests
import yaml

from plantseg import model_zoo_path, custom_zoo, home_path, PLANTSEG_MODELS_DIR, plantseg_global_path
from plantseg.__version__ import __version__ as current_version
from plantseg.pipeline import gui_logger

CONFIG_TRAIN_YAML = "config_train.yml"
BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"


def load_config(config_path: str) -> dict:
    """
    load a yaml config in a dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_model_zoo() -> dict:
    """
    returns a dictionary of all models in the model zoo.
    example:
        {
        ...
        generic_confocal_3d_unet:
            path: 'download link or model location'
            resolution: [0.235, 0.150, 0.150]
            description: 'Unet trained on confocal images on 1/2-resolution in XY with BCEDiceLoss.'
        ...
        }
    """
    zoo_config = os.path.join(model_zoo_path)

    zoo_config = load_config(zoo_config)

    custom_zoo_config = load_config(custom_zoo)

    if custom_zoo_config is None:
        custom_zoo_config = {}

    zoo_config.update(custom_zoo_config)
    return zoo_config


def list_models(dimensionality_filter: list[str] = None,
                modality_filter: list[str] = None,
                output_type_filter: list[str] = None) -> list[str]:
    """
    return a list of models in the model zoo by name
    """
    zoo_config = get_model_zoo()
    models = list(zoo_config.keys())

    if dimensionality_filter is not None:
        models = [model for model in models if zoo_config[model].get('dimensionality', None) in dimensionality_filter]

    if modality_filter is not None:
        models = [model for model in models if zoo_config[model].get('modality', None) in modality_filter]

    if output_type_filter is not None:
        models = [model for model in models if zoo_config[model].get('output_type', None) in output_type_filter]

    return models


def get_model_description(model_name: str) -> str:
    """
    return the description of a model
    """
    zoo_config = get_model_zoo()
    if model_name not in zoo_config:
        raise ValueError(f'Model {model_name} not found in the model zoo.')

    description = zoo_config[model_name].get('description', None)
    if description is None or description == '':
        return 'No description available for this model.'

    return description


def _list_all_metadata(metadata_key: str) -> list[str]:
    """
    return a list of all properties in the model zoo
    """
    zoo_config = get_model_zoo()
    properties = list(set([zoo_config[model].get(metadata_key, None) for model in zoo_config]))
    properties = [prop for prop in properties if prop is not None]
    return sorted(properties)


def list_all_dimensionality() -> list[str]:
    """
    return a list of all dimensionality in the model zoo
    """
    return _list_all_metadata('dimensionality')


def list_all_modality() -> list[str]:
    """
    return a list of all modality in the model zoo
    """
    return _list_all_metadata('modality')


def list_all_output_type() -> list[str]:
    """
    return a list of all output_type in the model zoo
    """
    return _list_all_metadata('output_type')


def get_model_resolution(model: str) -> list[float, float, float]:
    """
    return a models reference resolution
    """
    zoo_config = get_model_zoo()
    resolution = zoo_config[model].get('resolution', [1., 1., 1.])
    return resolution


def add_custom_model(new_model_name: str,
                     location: Path = Path.home(),
                     resolution: Tuple[float, float, float] = (1., 1., 1.),
                     description: str = '',
                     dimensionality: str = '',
                     modality: str = '',
                     output_type: str = '') -> Tuple[bool, Optional[str]]:
    """
    Add a custom trained model in the model zoo
    :param new_model_name: name of the new model
    :param location: location of the directory containing the custom trained model
    :param resolution: reference resolution of the custom trained model
    :param description: description of the trained model
    :param dimensionality: dimensionality of the trained model
    :param modality: modality of the trained model
    :param output_type: output type of the trained model
    :return:
    """

    dest_dir = os.path.join(home_path, PLANTSEG_MODELS_DIR, new_model_name)
    os.makedirs(dest_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(location, "*"))
    all_expected_files = ['config_train.yml',
                          'last_checkpoint.pytorch',
                          'best_checkpoint.pytorch']

    recommended_patch_size = [80, 170, 170]
    for file in all_files:
        if os.path.basename(file) == 'config_train.yaml':
            config_train = load_config(file)
            recommended_patch_size = list(config_train['loaders']['train']['slice_builder']['patch_shape'])

        if os.path.basename(file) in all_expected_files:
            copy2(file, dest_dir)
            all_expected_files.remove(os.path.basename(file))

    if len(all_expected_files) != 0:
        msg = f'It was not possible to find in the directory specified {all_expected_files}, ' \
              f'the model can not be loaded.'
        return False, msg

    custom_zoo_dict = load_config(custom_zoo)
    if custom_zoo_dict is None:
        custom_zoo_dict = {}

    custom_zoo_dict[new_model_name] = {}
    custom_zoo_dict[new_model_name]["path"] = str(location)
    custom_zoo_dict[new_model_name]["resolution"] = resolution
    custom_zoo_dict[new_model_name]["description"] = description
    custom_zoo_dict[new_model_name]["recommended_patch_size"] = recommended_patch_size
    custom_zoo_dict[new_model_name]["dimensionality"] = dimensionality
    custom_zoo_dict[new_model_name]["modality"] = modality
    custom_zoo_dict[new_model_name]["output_type"] = output_type

    with open(custom_zoo, 'w') as f:
        yaml.dump(custom_zoo_dict, f)

    return True, None


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
    train_config_path = os.path.join(home_path,
                                     PLANTSEG_MODELS_DIR,
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
        model_dir = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, model_name)
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
        model_file = os.path.join(plantseg_global_path, "resources", "models_zoo.yaml")
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
            ps_models_dir = os.path.join(home_path, PLANTSEG_MODELS_DIR)
            shutil.rmtree(ps_models_dir)
            print("All models deleted... PlantSeg will now close")
            return None

        elif answer == 'n':
            print("Nothing was deleted.")
            return None

        else:
            print("Invalid input, please type 'y' or 'n'.")


def check_version(plantseg_url=' https://api.github.com/repos/hci-unihd/plant-seg/releases/latest'):
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
