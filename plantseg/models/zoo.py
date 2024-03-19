"""Model Zoo Singleton"""

from pathlib import Path
from typing import Dict, Optional

from plantseg import model_zoo_path, custom_zoo_path
from plantseg.utils import load_config


class ModelZooRecord:
    """A record in the PlantSeg model zoo.

    Example YAML Record:

    PlantSeg_3Dnuc_platinum:
        model_url: https://zenodo.org/records/10070349/files/FOR2581_PlantSeg_Plant_Nuclei_3D.pytorch
        resolution: [0.2837 0.1268 0.1268]
        description: "A generic 3D U-Net trained to predict the nuclei and their boundaries in plant. Voxel size: (0.1268×0.1268×0.2837 µm^3) (XYZ)"
        dimensionality: "3D"
        modality: "confocal"
        recommended_patch_size: [128, 256, 256]
        output_type: "nuclei"
    """

    def __init__(self, model_url, resolution, description, dimensionality, modality, recommended_patch_size, output_type, added_by):
        self.model_url = model_url
        self.resolution = resolution
        self.description = description
        self.dimensionality = dimensionality
        self.modality = modality
        self.recommended_patch_size = recommended_patch_size
        self.output_type = output_type
        self.added_by = added_by


class ModelZoo:
    _instance: Optional['ModelZoo'] = None
    _models: Dict
    # _models: Dict[str, ModelZooRecord]

    def __new__(cls, path_zoo, path_zoo_custom):
        if cls._instance is None:
            cls._instance = super(ModelZoo, cls).__new__(cls)
            cls._instance._initialisation(path_zoo, path_zoo_custom)
        return cls._instance

    def _initialisation(self, path_zoo, path_zoo_custom):
        self._models = self._get_model_zoo(path_zoo, path_zoo_custom)

    def _get_model_zoo(self, path_zoo: str | Path, path_zoo_custom: Optional[str | Path] = None) -> dict:
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
        config_zoo = load_config(Path(path_zoo))

        if path_zoo_custom is not None:
            config_zoo_custom = load_config(Path(path_zoo_custom))
            if config_zoo_custom is None:
                config_zoo_custom = {}

            config_zoo.update(config_zoo_custom)
        return config_zoo

    def get_model_zoo(self, get_custom: bool = True) -> dict:
        return self._models

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
        zoo_config = self._models
        models = list(zoo_config.keys())

        if dimensionality_filter is not None:
            models = [model for model in models if zoo_config[model].get('dimensionality', None) in dimensionality_filter]

        if modality_filter is not None:
            models = [model for model in models if zoo_config[model].get('modality', None) in modality_filter]

        if output_type_filter is not None:
            models = [model for model in models if zoo_config[model].get('output_type', None) in output_type_filter]

        return models

    def register_model(
        self,
        name,
        model_url,
        resolution,
        description,
        dimensionality,
        modality,
        recommended_patch_size,
        output_type,
        added_by,
    ):
        self._models[name] = ModelZooRecord(
            model_url,
            resolution,
            description,
            dimensionality,
            modality,
            recommended_patch_size,
            output_type,
            added_by,
        )

    def get_model(self, name):
        return self._models[name]

    def get_model_names(self):
        return list(self._models.keys())

    def get_model_description(self, model_name: str) -> str:
        """
        return the description of a model
        """
        zoo_config = self._models
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


model_zoo = ModelZoo(model_zoo_path, custom_zoo_path)
