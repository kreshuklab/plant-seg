import numpy as np


def recommend_patch_and_halo_shapes_3d(
    full_volume_shape: tuple[int, int, int],
    max_patch_shape: tuple[int, int, int],
    min_halo_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Recommend patch shape and halo size for a given 3D sample shape.

    Args:
        full_volume_shape (tuple[int, int, int]): Shape of the entire 3D sample.
        max_patch_shape (tuple[int, int, int]): Maximum feasible patch shape for the selected GPU.
        min_halo_shape (tuple[int, int, int]): Minimum halo size, counting both sides.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]: Recommended patch shape and halo shape.
    """
    full_volume_shape = np.array(full_volume_shape)
    max_patch_shape = np.array(max_patch_shape)
    min_halo_shape = np.array(min_halo_shape)

    n_voxels_patch = np.prod(max_patch_shape)
    n_voxels_sample = np.prod(full_volume_shape)

    if n_voxels_patch >= n_voxels_sample:
        return tuple(full_volume_shape), (0, 0, 0)

    # Adjust patch shape if necessary
    shrink = max_patch_shape > full_volume_shape
    adjusted_patch_shape = np.minimum(full_volume_shape, max_patch_shape)

    if np.any(shrink):
        n_shrink_dims = np.sum(shrink)
        if n_shrink_dims == 2:
            remaining_dim = np.flatnonzero(~shrink)
            adjusted_patch_shape[remaining_dim] = min(
                n_voxels_patch // np.prod(full_volume_shape[shrink]), full_volume_shape[remaining_dim]
            )
        elif n_shrink_dims == 1:
            remaining_dims = np.flatnonzero(~shrink)
            max_area = n_voxels_patch // full_volume_shape[shrink]
            if max_area <= np.prod(full_volume_shape[remaining_dims]):
                adjusted_patch_shape[remaining_dims] = int(np.sqrt(max_area))
            else:
                return tuple(full_volume_shape), (0, 0, 0)
    halo_shape = np.where(shrink, 0, min_halo_shape)
    return tuple(adjusted_patch_shape - halo_shape), tuple(halo_shape)


if __name__ == "__main__":
    sample_shapes = [
        ((95, 120, 500), (192, 192, 192), (88, 88, 88)),
        ((95, 120, 5000), (192, 192, 192), (88, 88, 88)),
        ((5000, 120, 95), (192, 192, 192), (88, 88, 88)),
        ((100, 1000, 1000), (192, 192, 192), (88, 88, 88)),
        ((1000, 1000, 1000), (192, 192, 192), (88, 88, 88)),
    ]
    for full_volume_shape, max_patch_shape, min_halo_shape in sample_shapes:
        result = recommend_patch_and_halo_shapes_3d(full_volume_shape, max_patch_shape, min_halo_shape)
        print(f"Input: {full_volume_shape, max_patch_shape, min_halo_shape} -> Output: {result}")
