import logging

import numpy as np
import torch
from torch import nn

from plantseg.training.model import UNet2D

logger = logging.getLogger(__name__)


def _is_2d_model(model: nn.Module) -> bool:
    if isinstance(model, nn.DataParallel):
        model = model.module
    return isinstance(model, UNet2D)


def find_patch_and_halo_shapes(
    full_volume_shape: tuple[int, int, int],
    max_patch_shape: tuple[int, int, int],
    min_halo_shape: tuple[int, int, int],
    both_sides: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Recommend patch shape and halo size for a given 3D sample shape.

    Args:
        full_volume_shape (tuple[int, int, int]): Shape of the entire 3D sample.
        max_patch_shape (tuple[int, int, int]): Maximum feasible patch shape for the selected GPU.
        min_halo_shape (tuple[int, int, int]): Minimum halo size, counting one side by default.
        both_sides (bool, optional): Whether the halo size is counted on both sides. Defaults to False.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]: Recommended patch shape and 1-side halo shape.
    """
    shape_volume = np.array(full_volume_shape)
    shape_patch_max = np.array(max_patch_shape)
    shape_halo_min = np.array(min_halo_shape) // 2 if both_sides else np.array(min_halo_shape)

    n_voxels_patch = np.prod(shape_patch_max)
    n_voxels_sample = np.prod(shape_volume)

    if n_voxels_patch >= n_voxels_sample:
        return tuple(shape_volume), (0, 0, 0)

    # Adjust patch shape if necessary
    shrink = shape_patch_max > shape_volume
    adjusted_patch_shape = np.minimum(shape_volume, shape_patch_max)

    if np.any(shrink):
        n_shrink_dims = np.sum(shrink)
        if n_shrink_dims == 2:
            remaining_dim = np.flatnonzero(~shrink)
            adjusted_patch_shape[remaining_dim] = min(
                n_voxels_patch // np.prod(shape_volume[shrink]), shape_volume[remaining_dim]
            )
        elif n_shrink_dims == 1:
            remaining_dims = np.flatnonzero(~shrink)
            max_area = n_voxels_patch // shape_volume[shrink]
            if max_area <= np.prod(shape_volume[remaining_dims]):
                adjusted_patch_shape[remaining_dims] = np.sqrt(max_area).astype(int)
            else:
                return tuple(shape_volume), (0, 0, 0)
    halo_shape = np.where(shrink, 0, shape_halo_min)
    return tuple(adjusted_patch_shape - halo_shape * 2), tuple(halo_shape)


def find_a_max_patch_shape(
    model: nn.Module,
    in_channels: int,
    device: str,
    # ) -> tuple[int, int, int] | tuple[int, int]:
) -> tuple[int, int, int]:
    """Determine the maximum feasible patch shape for a given model based on available GPU memory using binary search.

    1. This is merely a good quick guess. However, an exact maximum size causes problems.
    2. If isotropic shape, then the `best_n` is 128 for 2080 Ti, 186 for 3090,
       but for the sake of speed, cap it at 50 and let batch size handle the rest.
    """

    nn_dim = 2 if _is_2d_model(model) else 3

    if device == "cpu":
        return (1, 1024, 1024) if nn_dim == 2 else (256, 256, 256)

    model = model.to(device)
    model.eval()

    if nn_dim == 3:  # use binary search for 3D
        low, high = (2, 50)
        best_n = low

        with torch.no_grad():
            while low <= high:
                mid = (low + high) // 2
                patch_shape = (16 * mid,) * nn_dim
                try:
                    x = torch.randn((1, in_channels) + patch_shape).to(device)
                    _ = model(x)
                    best_n = mid  # Update best_n if successful
                    low = mid + 1  # Try larger patches
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        high = mid - 1  # Try smaller patches
                    else:
                        raise
                finally:
                    del x
                    torch.cuda.empty_cache()

    else:  # use linear search for 2D
        best_n = 200

        with torch.no_grad():
            while best_n > 16:
                patch_shape = (16 * best_n,) * nn_dim
                try:
                    x = torch.randn((1, in_channels) + patch_shape).to(device)
                    _ = model(x)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        best_n -= 20
                    else:
                        raise
                finally:
                    del x
                    torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    return (1, 16 * best_n, 16 * best_n) if nn_dim == 2 else (16 * best_n, 16 * best_n, 16 * best_n)


def find_batch_size(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    device: str,
) -> int:
    """Determine the maximum feasible batch size for a given model based on available GPU memory.

    Args:
        model (nn.Module): The model to be used for predictions.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        device (str): Device to perform the computation on ('cuda' or 'cpu').

    Returns:
        int: The largest batch size that can be used without causing memory overflow.
    """
    if device == "cpu":
        return 1

    actual_patch_shape = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))
    if isinstance(model, UNet2D):
        actual_patch_shape = actual_patch_shape[1:]

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                x = torch.randn((batch_size, in_channels) + actual_patch_shape).to(device)
                _ = model(x)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size //= 2
                    break
                else:
                    raise
            finally:
                del x
                torch.cuda.empty_cache()
    if batch_size == 0:
        raise RuntimeError(
            f"Could not determine a feasible batch size for patch size {patch_shape} and halo {patch_halo}. "
            "Please reduce the patch size."
        )
    del model
    torch.cuda.empty_cache()
    return batch_size


def will_CUDA_OOM(
    model: nn.Module,
    in_channels: int,
    patch_shape: tuple[int, int, int],
    patch_halo: tuple[int, int, int],
    batch_size: int,
    device: str,
) -> bool:
    """Determine if a given batch size will cause an out-of-memory (OOM) error on the specified device.

    Args:
        model (nn.Module): The model to be used for predictions.
        in_channels (int): Number of input channels to the model.
        patch_shape (tuple[int, int, int]): Dimensions of the input patches.
        patch_halo (tuple[int, int, int]): Halo size used for patch augmentation.
        batch_size (int): Number of samples per batch.
        device (str): Device to perform the computation on ('cuda' or 'cpu').

    Returns:
        bool: True if the batch size will cause an OOM error, False otherwise.
    """
    if device == "cpu":
        return False  # CPU does not have CUDA OOM errors

    # Calculate the actual patch shape including the halo
    actual_patch_shape = tuple(patch_shape[i] + 2 * patch_halo[i] for i in range(3))

    # If the model is a 2D UNet, adjust the patch shape to 2D
    if isinstance(model, UNet2D):
        actual_patch_shape = actual_patch_shape[1:]

    model = model.to(device)
    model.eval()

    OOM_error = False
    x = None
    try:
        with torch.no_grad():
            x = torch.randn((batch_size, in_channels) + actual_patch_shape).to(device)
            _ = model(x)
    except RuntimeError as e:
        if "out of memory" in str(e):
            OOM_error = True
            logger.info(
                f"Using patch shape {patch_shape}, halo {patch_halo}, and batch size {batch_size} will cause OOM."
            )
        else:
            raise  # Re-raise if it's not an OOM error
    finally:
        del x
        del model
        torch.cuda.empty_cache()

    return OOM_error
