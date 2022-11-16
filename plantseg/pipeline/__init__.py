import logging
import os
import sys

from plantseg import plantseg_global_path

gui_logger = logging.getLogger("PlantSeg")
# hardcode the log-level for now
gui_logger.setLevel(logging.INFO)

# Add console handler (should show in GUI and on the console)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
gui_logger.addHandler(stream_handler)

# Resources directory
RESOURCES_DIR = "resources"
raw2seg_config_template = os.path.join(plantseg_global_path, RESOURCES_DIR, "raw2seg_template.yaml")
