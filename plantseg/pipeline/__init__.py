import logging

import sys

gui_logger = logging.getLogger("PlantSeg")
# hardcode the log-level for now
gui_logger.setLevel(logging.INFO)

# Add console handler (should show in GUI and on the console)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
gui_logger.addHandler(stream_handler)

# allowed h5 keys
H5_KEYS = ["raw", "predictions", "segmentation"]