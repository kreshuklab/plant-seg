from pathlib import Path

import yaml

# Find the global path of  plantseg
PLANTSEG_GLOBAL_PATH = Path(__file__).parent.absolute()

# Create configs directory at startup
USER_HOME_PATH = Path.home()

PLANTSEG_LOCAL_DIR = USER_HOME_PATH / '.plantseg'
PLANTSEG_MODELS_DIR = PLANTSEG_LOCAL_DIR / 'models'
PLANTSEG_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# create a user zoo config if does not exist
USER_MODEL_ZOO_CONFIG = PLANTSEG_MODELS_DIR / 'user_model_zoo.yaml'

if not USER_MODEL_ZOO_CONFIG.exists():
    with open(USER_MODEL_ZOO_CONFIG, 'w') as f:
        yaml.dump({}, f)

CONFIGS_PATH = PLANTSEG_LOCAL_DIR / 'configs'
CONFIGS_PATH.mkdir(parents=True, exist_ok=True)

# create a custom datasets config if does not exist
USER_DATASETS_CONFIG = PLANTSEG_LOCAL_DIR / 'user_datasets.yaml'

if not USER_DATASETS_CONFIG.exists():
    with open(USER_DATASETS_CONFIG, 'w') as f:
        yaml.dump({}, f)

# Resources directory
RESOURCES_DIR = 'resources'
MODEL_ZOO_PATH = PLANTSEG_GLOBAL_PATH / RESOURCES_DIR / 'models_zoo.yaml'
STANDARD_CONFIG_TEMPLATE = PLANTSEG_GLOBAL_PATH / RESOURCES_DIR / 'config_gui_template.yaml'
