import multiprocessing

from plantseg.viewer.widget.predictions import ALL_DEVICES, ALL_CUDA_DEVICES
from plantseg.viewer.widget.training import widget_unet_training

all_gpus_str = f'all gpus: {len(ALL_CUDA_DEVICES)}'
ALL_GPUS = [all_gpus_str] if len(ALL_CUDA_DEVICES) > 0 else []
ALL_DEVICES_HEADLESS = ALL_DEVICES + ALL_GPUS

MAX_WORKERS = len(ALL_CUDA_DEVICES) if len(ALL_CUDA_DEVICES) > 0 else multiprocessing.cpu_count()


def run_training_headless():
    widget_unet_training.show(run=True)
