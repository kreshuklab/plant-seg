# PlantSeg Segmentation

In this section we will describe how to use the PlantSeg segmentation workflows from the python API.

## API-Reference: [plantseg.predictions.functional.segmentation](https://github.com/hci-unihd/plant-seg/blob/master/plantseg/segmentation/functional/segmentation.py)
* ***dt_watershed***
```python
def dt_watershed(boundary_pmaps: np.ndarray,
                 threshold: float = 0.5,
                 sigma_seeds: float = 1.,
                 stacked: bool = False,
                 sigma_weights: float = 2.,
                 min_size: int = 100,
                 alpha: float = 1.0,
                 pixel_pitch: tuple[int, ...] = None,
                 apply_nonmax_suppression: bool = False,
                 n_threads: int = None,
                 mask: np.ndarray = None) -> np.ndarray:
    """ Wrapper around elf.distance_transform_watershed
    Args:
        boundary_pmaps (np.ndarray): input height map.
        threshold (float): value for the threshold applied before distance transform.
        sigma_seeds (float): smoothing factor for the watershed seed map.
        stacked (bool): if true the ws will be executed in 2D slice by slice, otherwise in 3D.
        sigma_weights (float): smoothing factor for the watershed weight map (default: 2).
        min_size (int): minimal size of watershed segments (default: 100)
        alpha (float): alpha used to blend input_ and distance_transform in order to obtain the
            watershed weight map (default: .9)
        pixel_pitch (list-like[int]): anisotropy factor used to compute the distance transform (default: None)
        apply_nonmax_suppression (bool): whether to apply non-maximum suppression to filter out seeds.
            Needs nifty. (default: False)
        n_threads (int): if not None, parallelize the 2D stacked ws. (default: None)
        mask (np.ndarray)
    Returns:
        np.ndarray: watershed segmentation
    """

    ...
```

https://github.com/hci-unihd/plant-seg/blob/de397694f523f142d67a38d5611acefd03e33137/plantseg/segmentation/functional/segmentation.py#L23-L122
* ***gasp***
```python
def gasp(boundary_pmaps: np.ndarray,
         superpixels: np.ndarray = None,
         gasp_linkage_criteria: str = 'average',
         beta: float = 0.5,
         post_minsize: int = 100,
         n_threads: int = 6) -> np.ndarray:
    """
    Implementation of the GASP algorithm for segmentation from affinities.
    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions.
        superpixels (np.ndarray): superpixel segmentation. If None, GASP will be run from the pixels. (default: None)
        gasp_linkage_criteria (str): Linkage criteria for GASP. (default: 'average')
        beta (float): beta parameter for GASP. A small value will steer the segmentation towards under-segmentation.
        While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after GASP. (default: 100)
        n_threads (int): number of threads used for GASP. (default: 6)
    Returns:
        np.ndarray: GASP output segmentation
    """

    ...
```

* ***mutex_ws***
```python
def mutex_ws(boundary_pmaps: np.ndarray,
             superpixels: np.ndarray = None,
             beta: float = 0.5,
             post_minsize: int = 100,
             n_threads: int = 6) -> np.ndarray:
    """
    Wrapper around gasp with mutex_watershed as linkage criteria.
    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions. 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
            If None, GASP will be run from the pixels. (default: None)
        beta (float): beta parameter for GASP. A small value will steer the segmentation towards under-segmentation.
            While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after GASP. (default: 100)
        n_threads (int): number of threads used for GASP. (default: 6)
    Returns:
        np.ndarray: GASP output segmentation
    """

    ...
```

* ***multicut***
```python
def multicut(boundary_pmaps: np.ndarray,
             superpixels: np.ndarray,
             beta: float = 0.5,
             post_minsize: int = 50) -> np.ndarray:

    """
    Multicut segmentation from boundary predictions.
    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
            under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)
    Returns:
        np.ndarray: Multicut output segmentation
    """

    ...
```

* ***lifted_multicut_from_nuclei_segmentation***
```python
def lifted_multicut_from_nuclei_segmentation(boundary_pmaps: np.ndarray,
                                             nuclei_seg: np.ndarray,
                                             superpixels: np.ndarray,
                                             beta: float = 0.5,
                                             post_minsize: int = 50) -> np.ndarray:
    """
    Lifted Multicut segmentation from boundary predictions and nuclei segmentation.
    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        nuclei_seg (np.ndarray): Nuclei segmentation. Must have the same shape as boundary_pmaps.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
        under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)
    Returns:
        np.ndarray: Multicut output segmentation
    """

    ...
```