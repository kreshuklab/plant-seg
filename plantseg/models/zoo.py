"""Model Zoo Singleton"""

from pathlib import Path
from typing import List, Optional, Self

from pandas import DataFrame, concat
from pydantic import BaseModel, Field, model_validator

from plantseg import PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM
from plantseg.utils import load_config


class ModelZooRecord(BaseModel):
    name: str
    url: Optional[str] = Field(None, alias='model_url')
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
        if self.url is None and self.path is None and self.id is None:
            print(self)
            raise ValueError('One of url, path or id must be present')
        return self


class ModelZoo:
    _instance: Optional['ModelZoo'] = None

    _zoo_dict: dict = {}
    _zoo_custom_dict: dict = {}

    path_zoo: str | Path = PATH_MODEL_ZOO
    path_zoo_custom: str | Path = PATH_MODEL_ZOO_CUSTOM
    models: DataFrame

    def __new__(cls, path_zoo, path_zoo_custom):
        if cls._instance is None:
            cls._instance = super(ModelZoo, cls).__new__(cls)
            cls._instance.update(path_zoo, path_zoo_custom)
        return cls._instance

    def _init_zoo_dict(
        self,
        path_zoo: Optional[str | Path] = None,
        path_zoo_custom: Optional[str | Path] = None,
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

            # zoo_dict.update(zoo_custom_dict)
            self._zoo_custom_dict = zoo_custom_dict

    def _init_zoo_df(self) -> None:
        # DataFrame.from_dict(zoo_dict, orient='index', columns=list(ModelZooRecord.__annotations__.keys()))
        records = []
        for name, model in self._zoo_dict.items():
            model['name'] = name
            records.append(ModelZooRecord(**model, added_by='plantseg').model_dump())

        for name, model in self._zoo_custom_dict.items():
            model['name'] = name
            records.append(ModelZooRecord(**model, added_by='user').model_dump())

        self.models = DataFrame(
            records,
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index('name')

    def update(
        self,
        path_zoo: Optional[str | Path] = None,
        path_zoo_custom: Optional[str | Path] = None,
    ) -> None:
        self._init_zoo_dict(path_zoo, path_zoo_custom)
        self._init_zoo_df()

    def get_model_zoo(self, get_custom: bool = True) -> dict:
        """
        returns a dictionary of all models in the model zoo.
        example:
            {
            ...
            generic_confocal_3d_unet:
                path: 'download link or model location'
                resolution: [0.235, 0.150, 0.150]
                description: 'Unet trained on confocal images on 1/2-resolution in XY with BCEDiceLoss.'
            ...
            }
        """
        return self.models

    def list_models(
        self,
        dimensionality_filter: Optional[list[str]] = None,
        modality_filter: Optional[list[str]] = None,
        output_type_filter: Optional[list[str]] = None,
        use_custom_models: bool = True,
    ) -> list[str]:
        """
        return a list of models in the model zoo by name
        """
        zoo_config = self.models
        models = list(zoo_config.keys())

        if dimensionality_filter is not None:
            models = [model for model in models if zoo_config[model].get('dimensionality', None) in dimensionality_filter]

        if modality_filter is not None:
            models = [model for model in models if zoo_config[model].get('modality', None) in modality_filter]

        if output_type_filter is not None:
            models = [model for model in models if zoo_config[model].get('output_type', None) in output_type_filter]

        return models

    def register_model(self, model_record: ModelZooRecord) -> None:
        """Add model_record to the model zoo dataframe"""
        models_new = DataFrame(
            [model_record.model_dump()],
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index('name')
        self.models = concat([self.models, models_new], ignore_index=False)

    def get_model(self, name):
        return self.models[name]

    def get_model_names(self):
        return list(self.models.keys())

    def get_model_description(self, model_name: str) -> str:
        """
        return the description of a model
        """
        zoo_config = self.models
        if model_name not in zoo_config:
            raise ValueError(f'Model {model_name} not found in the model zoo.')

        description = zoo_config[model_name].get('description', None)
        if description is None or description == '':
            return 'No description available for this model.'

        return description

    def _list_all_metadata(self, metadata_key: str) -> list[str]:
        """
        return a list of all properties in the model zoo
        """
        properties = list(set([self.get_model(model_name).get(metadata_key, None) for model_name in self.get_model_names()]))
        properties = [prop for prop in properties if prop is not None]
        return sorted(properties)

    def list_all_dimensionality(self) -> list[str]:
        """
        return a list of all dimensionality in the model zoo
        """
        return self._list_all_metadata('dimensionality')

    def list_all_modality(self) -> list[str]:
        """
        return a list of all modality in the model zoo
        """
        return self._list_all_metadata('modality')

    def list_all_output_type(self) -> list[str]:
        """
        return a list of all output_type in the model zoo
        """
        return self._list_all_metadata('output_type')

    def get_model_resolution(self, model_name: str) -> list[float]:
        """
        return a models reference resolution
        """
        return self.get_model(model_name).resolution


model_zoo = ModelZoo(PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM)
