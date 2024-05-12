"""Model Zoo Singleton"""

from warnings import warn
from pathlib import Path
from shutil import copy2
from typing import List, Tuple, Optional, Self

from pandas import DataFrame, concat
from pydantic import BaseModel, Field, AliasChoices, model_validator

from plantseg import PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM, PATH_PLANTSEG_MODELS
from plantseg import FILE_CONFIG_TRAIN_YAML, FILE_BEST_MODEL_PYTORCH, FILE_LAST_MODEL_PYTORCH
from plantseg.utils import get_class, load_config, save_config, download_files

AUTHOR_PLANTSEG = 'plantseg'
AUTHOR_USER = 'user'


class ModelZooRecord(BaseModel):
    """Model Zoo Record"""

    name: str
    url: Optional[str] = Field(None, alias=AliasChoices('model_url', 'url'))  # type: ignore
    path: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    resolution: Optional[Tuple[float, float, float]] = None
    dimensionality: Optional[str] = None
    modality: Optional[str] = None
    recommended_patch_size: Optional[Tuple[float, float, float]] = None
    output_type: Optional[str] = None
    doi: Optional[str] = None
    added_by: Optional[str] = None

    @model_validator(mode='after')
    def check_one_id_present(self) -> Self:
        """Check that one of url (zenodo), path (custom/local) or id (bioimage.io) is present"""
        if self.url is None and self.path is None and self.id is None:
            raise ValueError(f'One of url, path or id must be present: {self}')
        return self


class ModelZoo:
    """Model Zoo Singleton

    A model zoo DataFrame serves as a central repository for model metadata.
    The DataFrame is indexed by model name and contains columns for model metadata.
    The ModelZoo class provides methods to update, query and add records to the DataFrame.

    Records are added as ModelZooRecord instances, validated by the ModelZooRecord class.
    """

    _instance: Optional['ModelZoo'] = None

    _zoo_dict: dict = {}
    _zoo_custom_dict: dict = {}

    path_zoo: Path = PATH_MODEL_ZOO
    path_zoo_custom: Path = PATH_MODEL_ZOO_CUSTOM

    models: DataFrame

    def __new__(cls, path_zoo, path_zoo_custom):
        if cls._instance is None:
            cls._instance = super(ModelZoo, cls).__new__(cls)
            cls._instance.refresh(path_zoo, path_zoo_custom)
        return cls._instance

    def _init_zoo_dict(
        self,
        path_zoo: Optional[Path] = None,
        path_zoo_custom: Optional[Path] = None,
    ) -> None:
        if path_zoo is not None:
            self.path_zoo = path_zoo
        if path_zoo_custom is not None:
            self.path_zoo_custom = path_zoo_custom

        zoo_dict = load_config(Path(self.path_zoo))
        self._zoo_dict = zoo_dict

        if self.path_zoo_custom is not None:
            zoo_custom_dict = load_config(Path(self.path_zoo_custom))
            if zoo_custom_dict is None:
                zoo_custom_dict = {}

            self._zoo_custom_dict = zoo_custom_dict

    def _init_zoo_df(self) -> None:
        records = []
        for name, model in self._zoo_dict.items():
            model['name'] = name
            records.append(ModelZooRecord(**model, added_by=AUTHOR_PLANTSEG).model_dump())

        for name, model in self._zoo_custom_dict.items():
            model['name'] = name
            records.append(ModelZooRecord(**model, added_by=AUTHOR_USER).model_dump())

        self.models = DataFrame(
            records,
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index('name')

    def refresh(
        self,
        path_zoo: Optional[Path] = None,
        path_zoo_custom: Optional[Path] = None,
    ) -> None:
        self._init_zoo_dict(path_zoo, path_zoo_custom)
        self._init_zoo_df()

    def get_model_zoo_dict(self) -> dict:
        return self.models.to_dict(orient='index')

    def list_models(
        self,
        use_custom_models: bool = True,
        modality_filter: Optional[List[str]] = None,
        output_type_filter: Optional[List[str]] = None,
        dimensionality_filter: Optional[List[str]] = None,
    ) -> List[str]:
        """Return a list of model names, filtered by the specified criteria"""
        filtered_df: DataFrame = self.models

        if dimensionality_filter is not None:
            # `loc` to avoid pylint E1136 ; following lines have no warning because filtered_df now is a DataFrame
            filtered_df = filtered_df.loc[filtered_df.loc[:, 'dimensionality'].isin(dimensionality_filter)]

        if modality_filter is not None:
            filtered_df = filtered_df[filtered_df['modality'].isin(modality_filter)]

        if output_type_filter is not None:
            filtered_df = filtered_df[filtered_df['output_type'].isin(output_type_filter)]

        if not use_custom_models:
            filtered_df = filtered_df[filtered_df['added_by'] != AUTHOR_USER]

        return filtered_df.index.tolist()

    def get_model_names(self) -> List[str]:
        return self.models.index.to_list()

    def _get_model_record(self, name) -> ModelZooRecord:
        return ModelZooRecord(name=name, **self.models.loc[name])

    def get_model_description(self, model_name: str) -> Optional[str]:
        return self._get_model_record(model_name).description

    def get_model_resolution(self, model_name: str) -> Optional[Tuple[float, float, float]]:
        return self._get_model_record(model_name).resolution

    def get_model_patch_size(self, model_name: str) -> Optional[Tuple[float, float, float]]:
        return self._get_model_record(model_name).recommended_patch_size

    def _get_unique_metadata(self, metadata_key: str) -> List[str]:
        metadata = self.models.loc[:, metadata_key].dropna().unique()
        return [str(x) for x in metadata]

    def get_unique_dimensionalities(self) -> List[str]:
        return self._get_unique_metadata('dimensionality')

    def get_unique_modalities(self) -> List[str]:
        return self._get_unique_metadata('modality')

    def get_unique_output_types(self) -> List[str]:
        return self._get_unique_metadata('output_type')

    def _add_model_record(self, model_record: ModelZooRecord) -> None:
        """Add a ModelZooRecord to the ModelZoo DataFrame"""
        models_new = DataFrame(
            [model_record.model_dump()],
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index('name')
        self.models = concat([self.models, models_new], ignore_index=False)

    def add_custom_model(
        self,
        new_model_name: str,
        location: Path = Path.home(),
        resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        description: str = '',
        dimensionality: str = '',
        modality: str = '',
        output_type: str = '',
    ) -> Tuple[bool, Optional[str]]:
        """Add a custom trained model in the model zoo local record file"""

        dest_dir = PATH_PLANTSEG_MODELS / new_model_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        all_expected_files = {
            FILE_CONFIG_TRAIN_YAML,
            FILE_LAST_MODEL_PYTORCH,
            FILE_BEST_MODEL_PYTORCH,
        }
        found_files = set()

        recommended_patch_size = [80, 170, 170]
        for file_path in location.glob("*"):
            if file_path.name == 'config_train.yml':
                try:
                    config_train = load_config(file_path)
                    recommended_patch_size = config_train['loaders']['train']['slice_builder']['patch_shape']
                except Exception as e:
                    return False, f'Failed to load or parse config_train.yml: {e}'

            if file_path.name in all_expected_files:
                copy2(file_path, dest_dir)
                found_files.add(file_path.name)

        missing_files = all_expected_files - found_files
        if missing_files:
            msg = f'Missing required files in the specified directory: {", ".join(missing_files)}. Model cannot be loaded.'
            return False, msg

        # Create and check the new model record
        new_model_record = {
            "path": str(location),
            "resolution": resolution,
            "description": description,
            "recommended_patch_size": recommended_patch_size,
            "dimensionality": dimensionality,
            "modality": modality,
            "output_type": output_type,
        }
        self._add_model_record(ModelZooRecord(name=new_model_name, **new_model_record, added_by=AUTHOR_USER))

        # Update the custom zoo dictionary in ModelZoo and save to file
        self._zoo_custom_dict[new_model_name] = new_model_record
        save_config(self._zoo_custom_dict, self.path_zoo_custom)

        return True, None

    def _download_model_files(self, model_url: str, out_dir: Path, config_only: bool = False) -> None:
        """Download model files and/or configuration based on the model URL."""
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        config_url = f"{model_url.rsplit('/', 1)[0]}/{FILE_CONFIG_TRAIN_YAML}"
        urls = {FILE_CONFIG_TRAIN_YAML: config_url}
        if not config_only:
            urls[FILE_BEST_MODEL_PYTORCH] = model_url
        download_files(urls, out_dir)

    def check_models(self, model_name: str, update_files: bool = False, config_only: bool = False) -> None:
        """Check and download model files and configurations as needed."""
        model_dir = PATH_PLANTSEG_MODELS / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Check if the model configuration file exists and download it if it doesn't
        if not (model_dir / FILE_CONFIG_TRAIN_YAML).exists() or update_files:
            model_file = PATH_MODEL_ZOO
            config = load_config(model_file)

            model_url = config.get(model_name, {}).get("model_url")
            if model_url:
                self._download_model_files(model_url, model_dir, config_only)
            else:
                warn(f"Model {model_name} not found in the models zoo configuration.")

    def _get_model_config_path_by_name(self, model_name: str) -> Path:
        """Return the path to the training configuration for a model in zoo."""
        self.check_models(model_name, config_only=True)
        return PATH_PLANTSEG_MODELS / model_name / FILE_CONFIG_TRAIN_YAML

    def get_model_config_by_name(self, model_name: str) -> dict:
        """Load the training configuration for a model in zoo."""
        config_path = self._get_model_config_path_by_name(model_name)
        return load_config(config_path)

    def _create_model_by_config(self, model_config: dict):
        """Create a model instance from a configuration."""
        model_class = get_class(model_config['name'], modules=['plantseg.training.model'])
        return model_class(**model_config)

    def get_model_by_config_path(self, config_path: Path, model_weights_path: Optional[Path] = None):
        """Create a safari model (may or may not be in zoo) from a configuration file."""
        config_train = load_config(config_path)
        model_config = config_train.pop('model')
        model = self._create_model_by_config(model_config)
        if model_weights_path is None:
            model_weights_path = config_path.parent / FILE_BEST_MODEL_PYTORCH
        return model, model_config, model_weights_path

    def get_model_by_name(self, model_name: str, model_update: bool = False):
        """Load configuration for a model in zoo; return the model, configuration and path."""
        self.check_models(model_name, update_files=model_update)
        config_path = self._get_model_config_path_by_name(model_name)
        model_weights_path = PATH_PLANTSEG_MODELS / model_name / FILE_BEST_MODEL_PYTORCH
        return self.get_model_by_config_path(config_path, model_weights_path)


model_zoo = ModelZoo(PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM)
