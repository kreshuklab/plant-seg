import logging
import sys

zoo_logger = logging.getLogger("PlantSeg Zoo")
# Hardcode the log-level for now
zoo_logger.setLevel(logging.INFO)

# Add console handler (should show in GUI and on the console)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
zoo_logger.addHandler(stream_handler)
