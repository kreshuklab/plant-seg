"""Initialise model registry at startup"""

from os import getenv
from pathlib import Path
import yaml


PATH_PLANTSEG_GLOBAL = Path(__file__).parent.resolve()

# Files in code repository
DIR_RESOURCES = "resources"
FILE_MODEL_ZOO = "models_zoo.yaml"
FILE_CONFIG_GUI_TEMPLATE = "config_gui_template.yaml"

PATH_MODEL_ZOO = PATH_PLANTSEG_GLOBAL / DIR_RESOURCES / FILE_MODEL_ZOO
PATH_STANDARD_TEMPLATE = PATH_PLANTSEG_GLOBAL / DIR_RESOURCES / FILE_CONFIG_GUI_TEMPLATE

# Files in user home
DIR_PLANTSEG_MODELS = ".plantseg_models"
DIR_CONFIGS = "configs"
FILE_MODEL_ZOO_CUSTOM = "custom_zoo.yaml"

PATH_HOME = Path(getenv('PLANTSEG_HOME', str(Path.home())))

PATH_CONFIGS = PATH_HOME / DIR_PLANTSEG_MODELS / DIR_CONFIGS
PATH_MODEL_ZOO_CUSTOM = PATH_HOME / DIR_PLANTSEG_MODELS / FILE_MODEL_ZOO_CUSTOM

PATH_CONFIGS.mkdir(parents=True, exist_ok=True)

if not PATH_MODEL_ZOO_CUSTOM.exists():
    with PATH_MODEL_ZOO_CUSTOM.open('w') as file:
        yaml.dump({}, file)
