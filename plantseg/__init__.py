import os
from pathlib import Path

# Find the global path of  plantseg
plantseg_global_path = Path(__file__).parent.absolute()

# Create configs directory at startup
home_path = os.path.expanduser("~")
configs_path = os.path.join(home_path, ".plantseg_models", "configs")
os.makedirs(configs_path, exist_ok=True)