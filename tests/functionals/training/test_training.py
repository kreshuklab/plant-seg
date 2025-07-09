import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from plantseg import FILE_CONFIG_TRAIN_YAML
from plantseg.functionals.training.train import (
    create_datasets,
    create_model_config,
    find_h5_files,
)


class TestFindH5Files:
    """Tests for find_h5_files function."""

    def test_find_h5_files_with_valid_directory(self):
        """Test finding h5 files in a valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files with different extensions
            test_files = [
                temp_path / "test1.h5",
                temp_path / "test2.hdf",
                temp_path / "test3.hdf5",
                temp_path / "test4.hd5",
                temp_path / "test5.txt",  # Should not be found
            ]

            for file in test_files:
                file.touch()

            found_files = find_h5_files(temp_path)

            # Should find 4 h5-type files
            assert len(found_files) == 4

            # Check that only h5-type files are found
            found_names = [f.name for f in found_files]
            assert "test1.h5" in found_names
            assert "test2.hdf" in found_names
            assert "test3.hdf5" in found_names
            assert "test4.hd5" in found_names
            assert "test5.txt" not in found_names

    def test_find_h5_files_empty_directory(self):
        """Test finding h5 files in an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            found_files = find_h5_files(temp_dir)
            assert len(found_files) == 0

    def test_find_h5_files_invalid_directory(self):
        """Test finding h5 files with invalid directory."""
        with pytest.raises(AssertionError):
            find_h5_files("/non/existent/directory")


class TestCreateModelConfig:
    """Tests for create_model_config function."""

    def test_create_model_config_2d(self):
        """Test creating model config for 2D UNet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"

            create_model_config(
                checkpoint_dir=checkpoint_dir,
                in_channels=1,
                out_channels=2,
                patch_size=[64, 64],
                dimensionality="2D",
                sparse=False,
                f_maps=[16, 32, 64],
                max_num_iters=1000,
            )

            # Check that directory was created
            assert checkpoint_dir.exists()

            # Check that config file was created
            config_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
            assert config_path.exists()

            # Check config content
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            assert config["model"]["in_channels"] == 1
            assert config["model"]["out_channels"] == 2
            assert config["model"]["f_maps"] == [16, 32, 64]
            assert config["model"]["name"] == "UNet2D"
            assert config["model"]["final_sigmoid"] is True  # not sparse
            assert config["trainer"]["checkpoint_dir"] == str(checkpoint_dir)
            assert config["trainer"]["max_num_iterations"] == 1000

    def test_create_model_config_3d(self):
        """Test creating model config for 3D UNet."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "test_checkpoint"

            create_model_config(
                checkpoint_dir=checkpoint_dir,
                in_channels=1,
                out_channels=3,
                patch_size=[32, 64, 64],
                dimensionality="3D",
                sparse=True,
                f_maps=[8, 16, 32],
                max_num_iters=2000,
            )

            config_path = checkpoint_dir / FILE_CONFIG_TRAIN_YAML
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            assert config["model"]["name"] == "UNet3D"
            assert config["model"]["final_sigmoid"] is False  # sparse
            assert config["trainer"]["max_num_iterations"] == 2000
            assert config["loaders"]["train"]["slice_builder"]["patch_shape"] == [
                32,
                64,
                64,
            ]
            assert config["loaders"]["train"]["slice_builder"]["stride_shape"] == [
                16,
                32,
                32,
            ]


class TestCreateDatasets:
    """Tests for create_datasets function."""

    def test_create_datasets_train(self):
        """Test creating training datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            train_dir = temp_path / "train"
            train_dir.mkdir()

            # Create dummy h5 files
            for i in range(3):
                h5_file = train_dir / f"data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            datasets = create_datasets(str(temp_path), "train", (8, 64, 64))

            assert len(datasets) == 3
            # Each dataset should be an HDF5Dataset instance
            from plantseg.functionals.training.h5dataset import HDF5Dataset

            for dataset in datasets:
                assert isinstance(dataset, HDF5Dataset)

    def test_create_datasets_val(self):
        """Test creating validation datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            datasets = create_datasets(str(temp_path), "val", (8, 64, 64))

            assert len(datasets) == 2

    def test_create_datasets_invalid_phase(self):
        """Test creating datasets with invalid phase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            val_dir = temp_path / "val"
            val_dir.mkdir()

            # Create dummy h5 files
            for i in range(2):
                h5_file = val_dir / f"val_data_{i}.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("raw", data=np.random.rand(10, 88, 88))
                    f.create_dataset("label", data=np.random.rand(10, 88, 88))

            with pytest.raises(AssertionError):
                create_datasets(str(temp_dir), "invalid_phase", (8, 64, 64))
