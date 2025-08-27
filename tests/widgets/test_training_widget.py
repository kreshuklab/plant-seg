from pathlib import Path

import pytest

from plantseg.functionals.training.train import find_h5_files
from plantseg.viewer_napari.widgets.training import Training_Tab


@pytest.fixture
def training_tab():
    return Training_Tab(None)


def test_get_container(training_tab):
    container = training_tab.get_container()
    assert len(container) == 3


def test_unet_training_run(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with("Starting training task", thread="train_gui")
    m_get_models.assert_not_called()
    m_schedule.assert_called_once()


def test_unet_training_no_dataset(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset=None,
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with("Please choose a dataset to load!", thread="train_gui")
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_no_image(
    training_tab,
    mocker,
    napari_segmentation,
    napari_raw,
):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="GUI",
        dataset=None,
        image=None,
        segmentation=napari_segmentation,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with(
        "Please choose a raw image and a segmentation to train!", thread="train_gui"
    )
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    training_tab.widget_unet_training(
        from_disk="GUI",
        dataset=None,
        image=napari_raw,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )
    m_log.assert_called_with(
        "Please choose a raw image and a segmentation to train!", thread="train_gui"
    )
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_none(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality=None,  # <-- error
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    m_log.reset_mock()
    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type=None,  # <-- error
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_custom(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality=training_tab.CUSTOM,  # <-- custom
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()

    m_log.reset_mock()
    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type=training_tab.CUSTOM,
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_no_name(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="",  # <-- error
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="confocal",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_not_called()


def test_unet_training_feature_maps(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_not_called()
    m_schedule.assert_called_once()
    assert isinstance(m_schedule.call_args[1]["task_kwargs"]["feature_maps"], int)

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained=None,
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16, 16, 16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_schedule.assert_called()
    assert isinstance(m_schedule.call_args[1]["task_kwargs"]["feature_maps"], list)


def test_unet_training_pretrained(training_tab, mocker):
    m_log = mocker.patch("plantseg.viewer_napari.widgets.training.log")
    m_get_models = mocker.patch(
        "plantseg.viewer_napari.widgets.training.model_zoo.get_model_by_name"
    )
    m_get_models.return_value = [1, 2, mocker.sentinel]
    m_schedule = mocker.patch("plantseg.viewer_napari.widgets.training.schedule_task")

    training_tab.widget_unet_training(
        from_disk="Disk",
        dataset="dataset/data",
        image=None,
        segmentation=None,
        pretrained="SOMETHING",  # <--
        model_name="test_model",
        description="description",
        channels=(1, 1),
        feature_maps=[16],
        patch_size=[16, 64, 64],
        resolution=[1.0, 1.0, 1.0],
        max_num_iters=100,
        dimensionality="3D",
        sparse=False,
        device="cpu",
        modality="boundaries",
        custom_modality="",
        output_type="boundaries",
        custom_output_type="",
        pbar=None,
    )

    m_log.assert_called_once()
    m_get_models.assert_called_with("SOMETHING")
    m_schedule.assert_called_once()
    assert m_schedule.call_args[1]["task_kwargs"]["pre_trained"] == mocker.sentinel


def test_on_from_disk_change(training_tab, mocker):
    training_tab.widget_unet_training.from_disk.value = "Disk"
    m_show_image = mocker.patch.object(training_tab.widget_unet_training.image, "show")
    m_show_seg = mocker.patch.object(
        training_tab.widget_unet_training.segmentation, "show"
    )
    m_show_dataset = mocker.patch.object(
        training_tab.widget_unet_training.dataset, "show"
    )
    m_hide_image = mocker.patch.object(training_tab.widget_unet_training.image, "hide")
    m_hide_seg = mocker.patch.object(
        training_tab.widget_unet_training.segmentation, "hide"
    )
    m_hide_dataset = mocker.patch.object(
        training_tab.widget_unet_training.dataset, "hide"
    )
    training_tab.widget_unet_training.from_disk.value = "GUI"

    m_show_image.assert_called_once()
    m_show_seg.assert_called_once()
    m_show_dataset.assert_not_called()

    m_hide_image.assert_not_called()
    m_hide_seg.assert_not_called()
    m_hide_dataset.assert_called_once()


def test_on_dimensionality_change(training_tab):
    training_tab.widget_unet_training.patch_size.value = [9, 8, 7]
    training_tab.widget_unet_training.dimensionality.value = "2D"
    assert training_tab.previous_patch_size == (9, 8, 7)
    assert training_tab.widget_unet_training.patch_size.value == (1, 8, 7)

    training_tab.widget_unet_training.dimensionality.value = "3D"
    assert (
        training_tab.widget_unet_training.patch_size.value
        == training_tab.previous_patch_size
    )


def test_on_custom_modality_change(training_tab, mocker):
    m_show = mocker.patch.object(
        training_tab.widget_unet_training.custom_modality, "show"
    )
    m_hide = mocker.patch.object(
        training_tab.widget_unet_training.custom_modality, "hide"
    )

    training_tab.widget_unet_training.modality.value = training_tab.CUSTOM
    m_show.assert_called_once()
    m_hide.assert_not_called()

    training_tab.widget_unet_training.modality.value = "confocal"
    m_hide.assert_called_once()
    m_show.assert_called_once()


def test_on_custom_output_type_change(training_tab, mocker):
    m_show = mocker.patch.object(
        training_tab.widget_unet_training.custom_output_type, "show"
    )
    m_hide = mocker.patch.object(
        training_tab.widget_unet_training.custom_output_type, "hide"
    )

    training_tab.widget_unet_training.output_type.value = training_tab.CUSTOM
    m_show.assert_called_once()
    m_hide.assert_not_called()

    training_tab.widget_unet_training.output_type.value = "boundaries"
    m_hide.assert_called_once()
    m_show.assert_called_once()


def test_on_dataset_change(training_tab, mocker):
    m_finder = mocker.patch(
        "plantseg.viewer_napari.widgets.training.find_h5_files",
        wraps=find_h5_files,
    )

    test_data_dir = Path(__file__).parent.parent / "resources" / "data"
    assert test_data_dir.exists(), f"Test data directory not found: {test_data_dir}"

    training_tab.widget_unet_training.dataset.value = test_data_dir / "Noneexistent"
    m_finder.assert_not_called()

    training_tab.widget_unet_training.dataset.value = test_data_dir.parent
    m_finder.assert_not_called()

    training_tab.widget_unet_training.dataset.value = test_data_dir
    m_finder.assert_called_once()
    assert training_tab.widget_unet_training.resolution.value == (0.28, 0.13, 0.13)


def test_on_pretrained_changed(training_tab, mocker):
    m_zoo = mocker.patch("plantseg.viewer_napari.widgets.training.model_zoo")
    m_zoo.get_model_by_name.return_value = ("model", {"f_maps": [16]}, "path")

    training_tab.widget_unet_training.pretrained.value = None
    m_zoo.get_model_by_name.assert_not_called()
    m_zoo.get_model_description.assert_not_called()

    training_tab.widget_unet_training.pretrained.choices = ["my_model"]
    training_tab.widget_unet_training.pretrained.value = "my_model"
    m_zoo.get_model_by_name.assert_called_with("my_model")
    m_zoo.get_model_description.assert_called_with("my_model")


def test_update_layer_selection(
    training_tab,
    mocker,
    napari_raw,
    napari_segmentation,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)

    sentinel = mocker.sentinel
    sentinel.value = napari_raw
    sentinel.type = "inserted"
    assert training_tab.widget_unet_training.image.value is None
    assert training_tab.widget_unet_training.segmentation.value is None

    training_tab.update_layer_selection(sentinel)
    assert napari_raw in training_tab.widget_unet_training.image.choices
    assert training_tab.widget_unet_training.segmentation.value is None

    viewer.add_layer(napari_segmentation)
    sentinel.value = napari_segmentation
    sentinel.type = "inserted"
    training_tab.update_layer_selection(sentinel)
    assert napari_segmentation in training_tab.widget_unet_training.segmentation.choices
