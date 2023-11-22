import os
from pathlib import Path

import yaml

# Find the global path of  plantseg
plantseg_global_path = Path(__file__).parent.absolute()

# Create configs directory at startup
if 'PLANTSEG_HOME' in os.environ:
    home_path = os.environ['PLANTSEG_HOME']
else:
    home_path = os.path.expanduser("~")
PLANTSEG_MODELS_DIR = ".plantseg_models"

configs_path = os.path.join(home_path, PLANTSEG_MODELS_DIR, "configs")
os.makedirs(configs_path, exist_ok=True)

# create custom zoo if does not exist
custom_zoo = os.path.join(home_path, PLANTSEG_MODELS_DIR, 'custom_zoo.yaml')

if not os.path.exists(custom_zoo):
    with open(custom_zoo, 'w') as f:
        yaml.dump({}, f)

# Resources directory
RESOURCES_DIR = "resources"
model_zoo_path = os.path.join(plantseg_global_path, RESOURCES_DIR, "models_zoo.yaml")
standard_config_template = os.path.join(plantseg_global_path, RESOURCES_DIR, "config_gui_template.yaml")
