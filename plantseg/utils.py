import shutil
from pathlib import Path
from warnings import warn

import requests
import yaml

from plantseg import PATH_HOME, DIR_PLANTSEG_MODELS, PATH_PLANTSEG_GLOBAL
from plantseg.pipeline import gui_logger

CONFIG_TRAIN_YAML = "config_train.yml"
BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"

def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file into a dictionary."""
    with config_path.open('r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def save_config(config: dict, config_path: Path) -> None:
    """Save a dictionary to a YAML configuration file."""
    with config_path.open('w') as file:
        yaml.dump(config, file)

def get_train_config(model_name: str) -> dict:
    """Load the training configuration for a specified model."""
    check_models(model_name, config_only=True)
    train_config_path = PATH_HOME / DIR_PLANTSEG_MODELS / model_name / CONFIG_TRAIN_YAML
    return load_config(train_config_path)

def download_files(urls: dict, out_dir: Path) -> None:
    """Download files from URLs to a specified directory."""
    for filename, url in urls.items():
        gui_logger.info(f"Downloading file {filename} from {url}...")
        try:
            response = requests.get(url, allow_redirects=True)
            response.raise_for_status()  # Raises HTTPError, if one occurred
            (out_dir / filename).write_bytes(response.content)
        except requests.RequestException as e:
            warn(f"Failed to download {url}. Error: {e}")

def manage_model_files(model_url: str, out_dir: Path, config_only: bool = False) -> None:
    """Download model files and/or configuration based on the model URL."""
    config_url = f"{model_url.rsplit('/', 1)[0]}/{CONFIG_TRAIN_YAML}"
    urls = {"config_train.yml": config_url}
    if not config_only:
        urls[BEST_MODEL_PYTORCH] = model_url
    download_files(urls, out_dir)

def check_models(model_name: str, update_files: bool = False, config_only: bool = False) -> None:
    """Check and download model files and configurations as needed."""
    model_dir = Path.home() / DIR_PLANTSEG_MODELS / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check if the model configuration file exists and download it if it doesn't
    if not (model_dir / CONFIG_TRAIN_YAML).exists() or update_files:
        model_file = PATH_PLANTSEG_GLOBAL / "resources" / "models_zoo.yaml"
        config = load_config(model_file)

        model_url = config.get(model_name, {}).get("model_url")
        if model_url:
            manage_model_files(model_url, model_dir, config_only)
        else:
            warn(f"Model {model_name} not found in the models zoo configuration.")

def clean_models() -> None:
    """Delete all models in the model zoo after confirmation from the user."""
    for _ in range(3):
        answer = input("This will delete all models in the model zoo. "
                       "Ensure you've backed up custom models. Continue? (y/n): ").lower()
        if answer == 'y':
            shutil.rmtree(PATH_HOME / DIR_PLANTSEG_MODELS, ignore_errors=True)
            print("All models deleted. PlantSeg will now close.")
            break
        elif answer == 'n':
            print("Operation cancelled. No models were deleted.")
            break
        else:
            print("Invalid input, please type 'y' or 'n'.")

def check_version(current_version: str, plantseg_url: str = 'https://api.github.com/repos/hci-unihd/plant-seg/releases/latest') -> None:
    """Check for the latest version of PlantSeg available on GitHub."""
    try:
        response = requests.get(plantseg_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
        latest_version = response.json()['tag_name']

        # Splitting the version string into components and comparing as tuples of integers
        if tuple(map(int, latest_version.strip('v').split('.'))) > tuple(map(int, current_version.strip('v').split('.'))):
            print(f"New version of PlantSeg available: {latest_version}. Please update to the latest version.")
        else:
            print(f"You are using the latest version of PlantSeg: {current_version}.")
    except requests.RequestException as e:
        warn(f"Could not check for new version. Error: {e}")
