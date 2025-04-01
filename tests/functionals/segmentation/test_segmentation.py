import numpy as np
import pytest

from plantseg.functionals.segmentation import dt_watershed

shapes = [(32, 64, 64), (64, 64)]
stacked_options = [True, False]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("stacked", stacked_options)
def test_dt_watershed(shape, stacked):
    mock_data = np.random.rand(*shape).astype("float32")

    if stacked and len(shape) == 2:
        with pytest.raises(ValueError):  # 2D data cannot be stacked
            dt_watershed(mock_data, stacked=stacked)
    else:
        result = dt_watershed(mock_data, stacked=stacked)
        assert isinstance(result, np.ndarray)
        assert result.shape == mock_data.shape
        assert result.dtype == np.uint64
        assert result.max() > result.min() >= 0
