"""Model Zoo Singleton"""

from pathlib import Path
from shutil import copy2
from typing import List, Tuple, Optional, Self

from pandas import DataFrame, concat
from pydantic import BaseModel, Field, AliasChoices, model_validator

from plantseg import PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM, PATH_HOME, DIR_PLANTSEG_MODELS
from plantseg.utils import load_config, save_config

AUTHOR_PLANTSEG = 'plantseg'
AUTHOR_USER = 'user'


class ModelZooRecord(BaseModel):
    """Model Zoo Record"""

    name: str
    url: Optional[str] = Field(None, alias=AliasChoices('model_url', 'url')) # type: ignore
    path: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    resolution: Optional[List[float]] = None
    dimensionality: Optional[str] = None
    modality: Optional[str] = None
    recommended_patch_size: Optional[List[int]] = None
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

    def get_model_resolution(self, model_name: str) -> Optional[List[float]]:
        return self._get_model_record(model_name).resolution

    def get_model_patch_size(self, model_name: str) -> Optional[List[int]]:
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

        dest_dir = PATH_HOME / DIR_PLANTSEG_MODELS / new_model_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        all_expected_files = {
            'config_train.yml',
            'last_checkpoint.pytorch',
            'best_checkpoint.pytorch',
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


model_zoo = ModelZoo(PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM)
