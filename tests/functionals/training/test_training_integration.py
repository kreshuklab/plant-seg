"""Integration tests for training."""

import tempfile
from pathlib import Path

import pytest
import torch

from panseg import FILE_MODEL_ZOO_CUSTOM, PATH_PANSEG_MODELS
from panseg.core.zoo import model_zoo
from panseg.functionals.training.train import unet_training


class TestUnetTrainingIntegration:
    """Integration tests for unet_training function using real H5 data."""

    def test_training_integration_3d_cpu(self, mocker):
        """Test actual training."""
        test_data_dir = Path(__file__).parent.parent.parent / "resources" / "data"
        assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

        train_dir = test_data_dir / "train"
        val_dir = test_data_dir / "val"
        assert train_dir.exists(), f"Train directory not found: {train_dir}"
        assert val_dir.exists(), f"Val directory not found: {val_dir}"

        train_files = list(train_dir.glob("*.h5"))
        val_files = list(val_dir.glob("*.h5"))
        assert len(train_files) > 0, f"No training H5 files found in {train_dir}"
        assert len(val_files) > 0, f"No validation H5 files found in {val_dir}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            model_name = "test_integration_3d_cpu"

            mocker.patch(
                "panseg.functionals.training.train.PATH_PANSEG_MODELS",
                temp_path,
            )
            mocker.patch.multiple(
                "panseg.core.zoo",
                PATH_PANSEG_MODELS=temp_path,
                PATH_MODEL_ZOO_CUSTOM=temp_path / FILE_MODEL_ZOO_CUSTOM,
            )
            mocker.patch.multiple(
                model_zoo,
                path_zoo=temp_path,
                path_zoo_custom=temp_path / FILE_MODEL_ZOO_CUSTOM,
            )
            unet_training(
                dataset_dir=str(test_data_dir),
                model_name=model_name,
                in_channels=1,
                out_channels=1,
                feature_maps=[8, 8],
                patch_size=(16, 64, 64),
                max_num_iters=10,
                dimensionality="3D",
                sparse=True,
                device="cpu",
            )

            model_dir = temp_path / model_name
            assert model_dir.exists(), f"Model directory not created: {model_dir}"

            checkpoint_files = list(model_dir.glob("*.pytorch"))
            assert len(checkpoint_files) > 0, "No checkpoint files created"

            config_file = model_dir / "config_train.yml"
            assert config_file.exists(), f"Config file not created: {config_file}"

            mocker.stopall()
            assert not (PATH_PANSEG_MODELS / model_name).exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_integration_3d_gpu(self, mocker):
        """Test actual training on GPU if available."""
        test_data_dir = Path(__file__).parent.parent.parent / "resources" / "data"
        assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

        train_dir = test_data_dir / "train"
        val_dir = test_data_dir / "val"
        assert train_dir.exists(), f"Train directory not found: {train_dir}"
        assert val_dir.exists(), f"Val directory not found: {val_dir}"

        train_files = list(train_dir.glob("*.h5"))
        val_files = list(val_dir.glob("*.h5"))
        assert len(train_files) > 0, f"No training H5 files found in {train_dir}"
        assert len(val_files) > 0, f"No validation H5 files found in {val_dir}"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            model_name = "test_integration_3d_gpu"

            mocker.patch(
                "panseg.functionals.training.train.PATH_PANSEG_MODELS",
                temp_path,
            )
            mocker.patch.multiple(
                "panseg.core.zoo",
                PATH_PANSEG_MODELS=temp_path,
                PATH_MODEL_ZOO_CUSTOM=temp_path / FILE_MODEL_ZOO_CUSTOM,
            )
            mocker.patch.multiple(
                model_zoo,
                path_zoo=temp_path,
                path_zoo_custom=temp_path / FILE_MODEL_ZOO_CUSTOM,
            )
            unet_training(
                dataset_dir=str(test_data_dir),
                model_name=model_name,
                in_channels=1,
                out_channels=1,
                feature_maps=16,
                patch_size=(16, 64, 64),
                max_num_iters=100,
                dimensionality="3D",
                sparse=False,
                device="cuda",
            )

            model_dir = temp_path / model_name
            assert model_dir.exists(), f"Model directory not created: {model_dir}"

            mocker.stopall()
            assert not (PATH_PANSEG_MODELS / model_name).exists()
