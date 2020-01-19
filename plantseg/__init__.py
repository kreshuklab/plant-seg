from pathlib import Path
import os
plantseg_global_path = Path(__file__).parent.absolute()

home_path = os.path.expanduser("~")
os.makedirs(home_path + "/.plantseg_models/configs", exist_ok=True)
