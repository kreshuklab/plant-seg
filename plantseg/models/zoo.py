"""Model Zoo Singleton"""

from pathlib import Path
from typing import List, Optional, Self

from pandas import DataFrame, concat
from pydantic import BaseModel, Field, model_validator

from plantseg import PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM
from plantseg.utils import load_config

AUTHOR_PLANTSEG = 'plantseg'
AUTHOR_USER = 'user'


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
            records.append(ModelZooRecord(**model, added_by=AUTHOR_PLANTSEG).model_dump())

        for name, model in self._zoo_custom_dict.items():
            model['name'] = name
            records.append(ModelZooRecord(**model, added_by=AUTHOR_USER).model_dump())

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

    def get_model_zoo_dict(self) -> dict:
        return self.models.to_dict(orient='index')

    def list_models(
        self,
        dimensionality_filter: Optional[List[str]] = None,
        modality_filter: Optional[List[str]] = None,
        output_type_filter: Optional[List[str]] = None,
        use_custom_models: bool = True,
    ) -> List[str]:
        """Return a list of model names, filtered by the specified criteria"""
        filtered_df = self.models

        if dimensionality_filter is not None:
            filtered_df = filtered_df[filtered_df['dimensionality'].isin(dimensionality_filter)]

        if modality_filter is not None:
            filtered_df = filtered_df[filtered_df['modality'].isin(modality_filter)]

        if output_type_filter is not None:
            filtered_df = filtered_df[filtered_df['output_type'].isin(output_type_filter)]

        if not use_custom_models:
            filtered_df = filtered_df[filtered_df['added_by'] != AUTHOR_USER]

        return filtered_df.index.tolist()

    def register_model(self, model_record: ModelZooRecord) -> None:
        """Add model_record to the model zoo dataframe"""
        models_new = DataFrame(
            [model_record.model_dump()],
            columns=list(ModelZooRecord.model_fields.keys()),
        ).set_index('name')
        self.models = concat([self.models, models_new], ignore_index=False)

    def get_model(self, name):
        return self.models.loc[name]

    def get_model_names(self):
        return self.models.index.to_list()

    def get_model_description(self, model_name: str) -> str:
        return self.get_model(model_name).description

    def get_model_resolution(self, model_name: str) -> list[float]:
        return self.get_model(model_name).resolution

    def _list_all_metadata(self, metadata_key: str) -> list[str]:
        metadata = self.models[metadata_key].dropna().unique()
        return [str(x) for x in metadata]

    def list_all_dimensionality(self) -> list[str]:
        return self._list_all_metadata('dimensionality')

    def list_all_modality(self) -> list[str]:
        return self._list_all_metadata('modality')

    def list_all_output_type(self) -> list[str]:
        return self._list_all_metadata('output_type')


model_zoo = ModelZoo(PATH_MODEL_ZOO, PATH_MODEL_ZOO_CUSTOM)
