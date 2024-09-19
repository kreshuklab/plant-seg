from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from plantseg import PATH_TRAINING_DATASETS
from plantseg.core import PlantSegImage
from plantseg.core.image import ImageDimensionality, ImageLayout, ImageType
from plantseg.io import H5_EXTENSIONS, create_h5

DATASET_SUFFIX = "dataset"


class TrainingParams(BaseModel):
    base_model: str


class Dataset(BaseModel):
    name: str
    directory: Path
    training_params: TrainingParams = TrainingParams(base_model="unet")
    image_layout: ImageLayout | None = None
    image_dimensionality: ImageDimensionality | None = None
    input_channels: int | None = None

    def model_post_init(self, __context) -> None:
        dir = Path(self.directory)
        dir.mkdir(parents=True, exist_ok=True)
        self.check_dataset()
        self.save()

    def save(self):
        save_dataset(self)

    @classmethod
    def load(cls, name: str):
        return load_dataset(name)

    def check_dataset(self):
        for file in self.directory.glob(f"*{H5_EXTENSIONS}"):
            pass

    def num_images(self):
        return len(list(self.directory.glob(f"*{H5_EXTENSIONS}")))

    def add_file(self, raw: PlantSegImage, label: PlantSegImage, mask: PlantSegImage = None):
        if self.image_layout is None:
            self.image_layout = raw.image_layout
        else:
            if self.image_layout != raw.image_layout:
                raise ValueError("Image layout must be the same for all images in the dataset")

        if self.image_dimensionality is None:
            self.image_dimensionality = raw.dimensionality
        else:
            if self.image_dimensionality != raw.dimensionality:
                raise ValueError("Image dimensionality must be the same for all images in the dataset")

        if raw.image_type is None:
            if raw.is_multichannel:
                self.input_channels = raw.data.shape[0]
            else:
                self.input_channels = 1
        else:
            if self.input_channels != raw.image_type.value:
                raise ValueError("Image type must be the same for all images in the dataset")

        create_dataset(self.directory, raw=raw, label=label, mask=mask)
        self.save()

    def list_images(self):
        return [f.stem for f in self.directory.glob(f"*{H5_EXTENSIONS}")]


def save_dataset(dataset: Dataset):
    dataset_dir = PATH_TRAINING_DATASETS
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = dataset_dir / f"{dataset.name}_{DATASET_SUFFIX}.yaml"
    with open(dataset_file, "w") as f:
        yaml.dump(dataset.model_dump(), f)


def load_dataset(name: str) -> Dataset:
    dataset_file = PATH_TRAINING_DATASETS / f"{name}_{DATASET_SUFFIX}.yaml"
    with open(dataset_file, "r") as f:
        dataset_dict = yaml.load(f, Loader=yaml.FullLoader)
    return Dataset(**dataset_dict)


def list_datasets():
    return [f.stem for f in PATH_TRAINING_DATASETS.glob(f"*{DATASET_SUFFIX}.yaml")]


def delete_dataset(name: str):
    dataset_file = PATH_TRAINING_DATASETS / f"{name}_{DATASET_SUFFIX}.yaml"
    dataset_file.unlink()


def create_dataset(dir: Path, raw: PlantSegImage, label: PlantSegImage, mask: PlantSegImage = None):
    def unique_name(dir: Path, name: str, suffix: str) -> Path:
        file_name = dir / f"{name}{suffix}"
        if not file_name.exists():
            return name

        i = 1
        while (dir / f"{name}_{i}{suffix}").exists():
            i += 1
        return dir / f"{name}_{i}{suffix}"

    file_name = unique_name(dir, raw.name, H5_EXTENSIONS)

    #  Check dimensionality
    if raw.dimensionality != label.dimensionality:
        raise ValueError("Raw and label images must have the same dimensionality")

    if mask is not None and raw.dimensionality != mask.dimensionality:
        raise ValueError("Raw and mask images must have the same dimensionality")

    # Check if shape is the same
    if raw.spatial_shape != label.spatial_shape:
        raise ValueError("Raw and label images must have the same shape")

    if mask.spatial_shape != raw.spatial_shape:
        raise ValueError("Raw and mask images must have the same shape")

    # Check type
    if label.image_type != ImageType.LABEL:
        raise ValueError("Label image must be of type LABEL")

    create_h5(path=file_name, stack=raw.data, key="raw", voxel_size=raw.voxel_size, mode="w")
    create_h5(path=file_name, stack=label.data, key="label", voxel_size=label.voxel_size, mode="a")
    if mask is not None:
        create_h5(path=file_name, stack=mask.data, key="mask", voxel_size=mask.voxel_size, mode="a")
