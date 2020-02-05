import numpy as np


def configure_segmentation(predictions_paths, config):
    name = config["name"]

    if name == "GASP" or name == "MutexWS" or name == "DtWatershed":
        from .gasp import GaspFromPmaps as Segmentation

    elif name == "MultiCut":
        from .multicut import MulticutFromPmaps as Segmentation

    else:
        raise NotImplementedError

    segmentation = Segmentation(predictions_paths)

    if name == "MutexWS":
        segmentation.__dict__["gasp_linkage_criteria"] = 'mutex_watershed'

    if name == "DtWatershed":
        segmentation.__dict__["gasp_linkage_criteria"] = 'DtWatershed'

    for name in segmentation.__dict__.keys():
        if name in config:
            segmentation.__dict__[name] = config[name]

    return segmentation


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