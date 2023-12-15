# PlantSeg CNN Predictions

In this section we will describe how to use the PlantSeg CNN Predictions workflow from the python API.

## API-Reference: [plantseg.predictions.functional.predictions](https://github.com/hci-unihd/plant-seg/blob/master/plantseg/predictions/functional/predictions.py)
### ***unet_predictions***
```python
def unet_predictions(raw: np.array,
                     model_name: str,
                     patch: Tuple[int, int, int] = (80, 160, 160),
                     stride: Union[str, Tuple[int, int, int]] = 'Accurate (slowest)',
                     device: str = 'cuda',
                     version: str = 'best',
                     model_update: bool = False,
                     mirror_padding: Tuple[int, int, int] = (16, 32, 32),
                     disable_tqdm: bool = False) -> np.array:
    """
    Predict boundaries predictions from raw data using a 3D U-Net model.

    Args:
        raw (np.array): raw data, must be a 3D array of shape (Z, Y, X) normalized between 0 and 1.
        model_name (str): name of the model to use. A complete list of available models can be found here:
        patch (tuple[int, int, int], optional): patch size to use for prediction. Defaults to (80, 160, 160).
        stride (Union[str, tuple[int, int, int]], optional): stride to use for prediction.
            If stride is defined as a string must be one of ['Accurate (slowest)', 'Balanced', 'Draft (fastest)'].
            Defaults to 'Accurate (slowest)'.
        device: (str, optional): device to use for prediction. Must be one of ['cpu', 'cuda', 'cuda:1', etc.].
            Defaults to 'cuda'.
        version (str, optional): version of the model to use, must be either 'best' or 'last'. Defaults to 'best'.
        model_update (bool, optional): if True will update the model to the latest version. Defaults to False.
        mirror_padding (tuple[int, int, int], optional): padding to use for prediction. Defaults to (16, 32, 32).
        disable_tqdm (bool, optional): if True will disable tqdm progress bar. Defaults to False.

    Returns:
        np.array: predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.

    """

    ...
```
```python
# Minimal example
from plantseg.predictions.functional.predictions import unet_predictions
import numpy as np

raw = np.random.random((128, 256, 256))
predictions = unet_predictions(raw,
                               model_name='generic_confocal_3d_unet',
                               patch=(80, 160, 160),
                               device='cuda')

```