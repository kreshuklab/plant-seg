SUPPORTED_ALGORITHMS = ["GASP", "MutexWS", "DtWatershed", "MultiCut", "LiftedMulticut", "SimpleITK"]


def configure_segmentation_step(predictions_paths, config):
    algorithm_name = config["name"]
    assert algorithm_name in SUPPORTED_ALGORITHMS, f"Unsupported algorithm name {algorithm_name}"

    # create a copy of the config to prevent changing the original
    config = config.copy()

    config['predictions_paths'] = predictions_paths

    if algorithm_name == "GASP":
        from .gasp import GaspFromPmaps

        # user 'average' linkage by default
        config['gasp_linkage_criteria'] = 'average'
        return GaspFromPmaps(**config)

    if algorithm_name == "MutexWS":
        from .gasp import GaspFromPmaps

        config['gasp_linkage_criteria'] = 'mutex_watershed'
        return GaspFromPmaps(**config)

    if algorithm_name == "DtWatershed":
        from .dtws import DistanceTransformWatershed

        return DistanceTransformWatershed(**config)

    if algorithm_name == "MultiCut":
        from .multicut import MulticutFromPmaps

        return MulticutFromPmaps(**config)

    if algorithm_name == "SimpleITK":
        from .simpleitkws import SimpleITKWatershed

        return SimpleITKWatershed(**config)

    if algorithm_name == "LiftedMulticut":
        from .lmc import LiftedMulticut

        assert (
            'nuclei_predictions_path' in config
        ), "Missing 'nuclei_predictions_path' config attribute for 'LiftedMulticut'"
        return LiftedMulticut(**config)
