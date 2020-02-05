# Data Processing 
PlantSeg includes basic utilities for data pre-processing and post-processing. 

## Pre-Procssing
![alt text](./images/preprocessing.png)

The input for this widget can be either a "raw" image or "prediction" image. 
Input format allowed are tiff and h5 while output is h5. \
The most important setting is the **rescaling factor**. It is very important to rescale the image to 
match the resolution of the data used for training the NeuralNetwork. \
```
As and example:
  - if your data has the voxel size of 0.3 x 0.1 x 0.1 (zxy).
  - and the networks was trained on 0.3 x 0.2 x 0.2 data (reference resolution).

The required voxel size can be obtained by computing the ratio between your data and the
reference at train. In the example the rescaling factor = 1 x 2 x 2. 
```
This operation can be done automatically by clicking in the gui on **auto re-scaling**. \
Be careful to use this function only in case of data considerably different from 
the reference resolution.

The **interpolation** field control the interpolation type (0 for nearest neighbors, 1 for linear spline, 
2 for quadratic).

The last field defines a **filters** operation. Implemented there are:
* **Gaussian** Filtering: The parameters is float and define the sigma value for the gaussian smoothing. 
The higher the wider is filtering kernel.

* **Median** Filtering: Apply median operation convolutionally over the image.
 The kernel is a sphere of size defined in the parameter field.

## Post-Procssing
![alt text](./images/postprocessing.png)

A post processing widget can be applied after the prediction and the segmentation.
Two type of operations can be performed:
 * Converting output to tiff file (default is h5). 
 * Rescaling the output back to the original resolution. 
