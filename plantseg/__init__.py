from pathlib import Path
import os
plantseg_global_path = Path(__file__).parent.absolute()

os.makedirs("~/.plantseg_models/configs", exist_ok=True)