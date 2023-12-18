# Classic Data Processing

![alt text](https://github.com/hci-unihd/plant-seg/raw/assets/images/preprocessing.png)
**PlantSeg** includes essential utilities for data pre-processing and post-processing.

## Pre-Processing

The input for this widget can be either a "raw" image or a "prediction" image.
Input formats allowed are tiff and h5, while output is always h5.

* **Save Directory** can be used to define the output directory.

* The most critical setting is the **Rescaling**. It is important to rescale the image to
 match the resolution of the data used for training the Neural Network.
This operation can be done automatically by clicking on the GUI on **Guided**.
Be careful to use this function only in case of data considerably different from
the reference resolution.
```
As an example:
  - if your data has the voxel size of 0.3 x 0.1 x 0.1 (ZYX).
  - and the networks was trained on 0.3 x 0.2 x 0.2 data (reference resolution).

The required voxel size can be obtained by computing the ratio between your data and the
reference train dataset. In the example the rescaling factor = 1 x 2 x 2.
```

* The **Interpolation** field controls the interpolation type (0 for nearest neighbors, 1 for linear spline,
2 for quadratic).

* The last field defines a **Filter** operation. Implemented there are:
    1. **Gaussian** Filtering: The parameter is a float and defines the sigma value for the gaussian smoothing.
The higher, the wider is filtering kernel.
    2. **Median** Filtering: Apply median operation convolutionally over the image.
 The kernel is a sphere of size defined in the parameter field.

## Post-Processing

A post-processing step can be performed after the **CNN-Predictions** and the **Segmentation**.
The post-processing options are:
 * Converting the output to the tiff file format (default is h5).

 * Casting the **CNN-Predictions** output to *data_uint8* drastically reduces the memory footprint of the output
 file.

Additionally, the post-processing will scale back your outputs to the original voxels resolutions.