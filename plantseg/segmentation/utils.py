import numpy as np

SUPPORTED_ALGORITMS = ["GASP", "MutexWS", "DtWatershed", "MultiCut"]


def configure_segmentation(predictions_paths, config):
    algorithm_name = config["name"]
    assert algorithm_name in SUPPORTED_ALGORITMS, f"Unsupported algorithm name {algorithm_name}"

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


def shift_affinities(affinities, offsets):
    rolled_affs = []
    for i, _ in enumerate(offsets):
        offset = offsets[i]
        shifts = tuple([int(off / 2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf

        padded_inverted_affs = np.pad(affinities, pad_width=((0, 0),) + tuple(padding), mode='constant')

        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1]) for ax in range(3))

        padded_inverted_affs = np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[crop_slices]
        rolled_affs.append(padded_inverted_affs)
        del padded_inverted_affs

    rolled_affs = np.stack(rolled_affs)
    return rolled_affs
