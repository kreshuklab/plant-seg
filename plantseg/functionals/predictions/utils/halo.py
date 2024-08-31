import numpy as np


def recommend_patch_and_halo_shapes_3d(
    shape_sample: tuple[int, int, int],
    shape_patch: tuple[int, int, int],
    shape_halo: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Recommend halo size for a given sample shape and patch shape.

    Args:
        shape_sample (tuple): Shape of the whole 3D sample.
        shape_patch (tuple): Shape of the practical maxinum patch for selected GPU.
        shape_halo (tuple): Shape of the theoretical minimum halo, count both sides.

    Returns:
        tuple: Recommended patch shape and halo shape.
    """
    shape_sample = np.array(shape_sample)
    shape_patch = np.array(shape_patch)
    shape_halo = np.array(shape_halo)

    n_voxels_patch = np.prod(shape_patch)
    n_voxels_sample = np.prod(shape_sample)

    if n_voxels_patch >= n_voxels_sample:  # sample can fit into GPU memory
        return tuple(shape_sample), (0, 0, 0)

    shrink = np.array(shape_patch) > np.array(shape_sample)
    if np.any(shrink):
        recommended_shape_total = np.array([s if b else p for s, p, b in zip(shape_sample, shape_patch, shrink)])
        if np.sum(shrink) == 2:
            d_max = n_voxels_patch / np.prod(shape_sample[shrink])
            if d_max > shape_sample[~shrink]:
                d_max = shape_sample[~shrink]
            recommended_shape_total[~shrink] = d_max
            recommended_shape_halo = shape_halo * (~shrink)
            return tuple(recommended_shape_total - recommended_shape_halo), tuple(recommended_shape_halo)
        elif np.sum(shrink) == 1:
            d_max_square = n_voxels_patch / shape_sample[shrink]
            if d_max_square > np.prod(shape_sample[~shrink]):
                return tuple(shape_sample), (0, 0, 0)
            else:
                recommended_shape_total[~shrink] = int(np.sqrt(d_max_square))
                recommended_shape_halo = shape_halo * (~shrink)
                return tuple(recommended_shape_total - recommended_shape_halo), tuple(recommended_shape_halo)
        else:
            return tuple(shape_sample), (0, 0, 0)

    else:
        return tuple(np.array(shape_sample) - np.array(shape_halo)), shape_halo


if __name__ == "__main__":
    shape_sample = (95, 120, 500)
    shape_patch = (192, 192, 192)
    shape_halo = (88, 88, 88)
    print(
        recommend_patch_and_halo_shapes_3d(shape_sample, shape_patch, shape_halo)
    )  # Expected: ((95, 120, 500), (0, 0, 0))

    shape_sample = (100, 1000, 1000)
    shape_patch = (192, 192, 192)
    shape_halo = (88, 88, 88)
    print(
        recommend_patch_and_halo_shapes_3d(shape_sample, shape_patch, shape_halo)
    )  # Expected: ((100, 178, 178), (0, 88, 88))

    shape_sample = (95, 120, 5000)
    shape_patch = (192, 192, 192)
    shape_halo = (88, 88, 88)
    print(
        recommend_patch_and_halo_shapes_3d(shape_sample, shape_patch, shape_halo)
    )  # Expected: ((95, 120, 532), (0, 0, 88))
