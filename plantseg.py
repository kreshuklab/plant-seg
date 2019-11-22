import argparse
import yaml
from plantseg import raw2seg


def _load_config():
    parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    # Load general configuration file
    config = _load_config()
    raw2seg(config)
