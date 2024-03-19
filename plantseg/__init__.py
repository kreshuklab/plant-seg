"""Initialise model registry at startup"""

from os import getenv
from pathlib import Path
import yaml


plantseg_global_path = Path(__file__).parent.resolve()

# Files in code repository
RESOURCES_DIR = "resources"
MODELS_ZOO_FILE_NAME = "models_zoo.yaml"
CONFIG_GUI_TEMPLATE_FILE_NAME = "config_gui_template.yaml"

model_zoo_path = plantseg_global_path / RESOURCES_DIR / MODELS_ZOO_FILE_NAME
standard_config_template = plantseg_global_path / RESOURCES_DIR / CONFIG_GUI_TEMPLATE_FILE_NAME

# Files in user home
PLANTSEG_MODELS_DIR = ".plantseg_models"
CONFIGS_DIR_NAME = "configs"
CUSTOM_ZOO_FILE_NAME = "custom_zoo.yaml"

home_path = Path(getenv('PLANTSEG_HOME', str(Path.home())))

configs_path = home_path / PLANTSEG_MODELS_DIR / CONFIGS_DIR_NAME
custom_zoo_path = home_path / PLANTSEG_MODELS_DIR / CUSTOM_ZOO_FILE_NAME

configs_path.mkdir(parents=True, exist_ok=True)

if not custom_zoo_path.exists():
    with custom_zoo_path.open('w') as file:
        yaml.dump({}, file)
