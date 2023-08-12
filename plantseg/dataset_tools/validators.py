from plantseg.dataset_tools.dataset_handler import DatasetHandler


class CheckDatasetDirectoryStructure:

    def __call__(self, dataset: DatasetHandler) -> tuple[bool, str]:
        # Check if dataset directory exists
        if not dataset.dataset_dir.exists():
            return False, f'Dataset directory {dataset.dataset_dir} does not exist.'

        # Check if dataset directory contains all expected subdirectories
        for phase in dataset.default_phases:
            if not (dataset.dataset_dir / phase).exists():
                return False, f'Dataset directory {dataset.dataset_dir} does not contain {phase} directory.'

        return True, ''
