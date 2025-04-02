import copy
import os

import yaml

arch = ["unet", "resunet"]
norm = ["bn", "gn"]
loss = ["bce", "dice", "bce_dice"]

arch_config_values = {
    "unet": {
        "model/name": "UNet3D",
        "loaders/train/slice_builder/patch_shape": [80, 170, 170],
        "loaders/val/slice_builder/patch_shape": [80, 170, 170],
        "loaders/val/slice_builder/stride_shape": [80, 170, 170],
        "loaders/test/slice_builder/patch_shape": [80, 170, 170],
    },
    "resunet": {
        "model/name": "ResidualUNet3D",
        "loaders/train/slice_builder/patch_shape": [80, 144, 144],
        "loaders/val/slice_builder/patch_shape": [80, 144, 144],
        "loaders/val/slice_builder/stride_shape": [80, 144, 144],
        "loaders/test/slice_builder/patch_shape": [80, 144, 144],
    },
}

norm_config_values = {
    "bn": {
        "model/layer_order": "bcr",
        "model/num_groups": None,
        "loaders/batch_size": 4,
        # this will work for both unet and resunet
        "loaders/train/slice_builder/patch_shape": [48, 80, 80],
        "loaders/val/slice_builder/patch_shape": [48, 80, 80],
        "loaders/val/slice_builder/stride_shape": [48, 80, 80],
        "loaders/test/slice_builder/patch_shape": [48, 80, 80],
        "loaders/test/slice_builder/stride_shape": [24, 40, 40],
    },
    "gn": {"model/layer_order": "gcr", "model/num_groups": 8},
}

loss_config_values = {
    "bce": {"loss/name": "BCEWithLogitsLoss"},
    "dice": {"loss/name": "DiceLoss"},
    "bce_dice": {"loss/name": "BCEDiceLoss"},
}

BASE_DIR = "/g/kreshuk/wolny/workspace/plant-seg/plantseg/resources/training_configs/"


def create_config(conf, output_file, a, n, l, phase):  # noqa: E741
    conf = copy.deepcopy(conf)
    updates = []
    updates.extend(arch_config_values[a].items())
    updates.extend(norm_config_values[n].items())
    if phase == "train":
        updates.extend(loss_config_values[l].items())
        # update checkpoint_dir
        checkpoint_dir = os.path.join(BASE_DIR, os.path.split(output_file)[0])
        updates.extend([("trainer/checkpoint_dir", checkpoint_dir)])
    if phase == "test":
        model_path = os.path.join(
            BASE_DIR, os.path.split(output_file)[0], "best_checkpoint.pytorch"
        )
        output_dir = os.path.join(
            BASE_DIR, os.path.split(output_file)[0], "predictions"
        )

        updates.extend([("model_path", model_path)])
        updates.extend([("loaders/output_dir", output_dir)])

    for key, value in updates:
        if phase == "test" and ("train" in key or "val" in key):
            continue
        if phase == "train" and "test" in key:
            continue

        path = key.split("/")
        conf_param = conf
        for p in path[:-1]:
            conf_param = conf_param[p]
        conf_param[path[-1]] = value

    yaml.safe_dump(conf, open(output_file, "w"))


def generate_configs():
    base_dir = "./grid_search"

    for phase in ["train", "test"]:
        for ds_name in ["root", "ovules"]:
            conf_file = os.path.join(base_dir, f"base_{phase}_config_{ds_name}.yml")
            conf = yaml.safe_load(open(conf_file, "r"))
            # generate train config
            for a in arch:
                for n in norm:
                    for l in loss:  # noqa: E741
                        output_dir = os.path.join(base_dir, ds_name, f"{a}_{n}_{l}")
                        output_path = os.path.join(output_dir, f"config_{phase}.yml")
                        os.makedirs(output_dir, exist_ok=True)
                        create_config(conf, output_path, a, n, l, phase)


if __name__ == "__main__":
    generate_configs()
