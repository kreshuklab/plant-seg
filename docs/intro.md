# PlantSeg introduction

PlantSeg is a tool for 3D and 2D segmentation.
The methods used are very generic and can be used for any instance segmentation workflow,
but they are tuned towards cell segmentation in plant tissue. The tool is fundamentally composed of two main steps.

![Main Figure](_images/main_figure.png)

* ***Cell boundary predictions***: Where a convolutional neural network is used to extract a
voxel wise boundary classification. The neural network can filter out very different types/intensities of
noise, homogenizing the signal strength and fixing imaging defects (such as missing/blurred cell boundaries).

* ***Cell Segmentation as graph partitioning***: The output of the first step can be used directly for automated
segmentation. We implemented four different algorithms for segmentation, each with peculiar features.
 This approach is especially well suited for segmenting densely packed cells.

For a complete description of the methods used, please check out our
[manuscript](https://elifesciences.org/articles/57613)

If you find PlantSeg useful, please cite {cite:p}`wolny2020accurate`.

```{bibliography}
:filter: docname in docnames
```
