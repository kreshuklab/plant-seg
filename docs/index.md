# PlantSeg introduction

PlantSeg is a tool for 3D and 2D segmentation.
The methods used are very generic and can be used for any instance segmentation workflow,
but they are tuned towards cell segmentation in plant tissue. The tool is fundamentally composed of two main steps.

![Main Figure](https://github.com/hci-unihd/plant-seg/raw/assets/images/main_figure.png)

* ***Cell boundary predictions***: Where a convolutional neural network is used to extract a
voxel wise boundary classification. The neural network can filter out very different types/intensities of
noise, homogenizing the signal strength and fixing imaging defects (such as missing/blurred cell boundaries).

* ***Cell Segmentation as graph partitioning***: The output of the first step can be used directly for automated
segmentation. We implemented four different algorithms for segmentation, each with peculiar features.
 This approach is especially well suited for segmenting densely packed cells.

For a complete description of the methods used, please check out our
[manuscript](https://elifesciences.org/articles/57613).

If you find PlantSeg useful, please cite:

```bibtex
@article{wolny2020accurate,
  title={Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
  author={Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, S{\"o}ren and Wilson-S{\'a}nchez, David and Lymbouridou, Rena and others},
  journal={Elife},
  volume={9},
  pages={e57613},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```
