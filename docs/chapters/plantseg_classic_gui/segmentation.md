# Segmentation

The segmentation widget allows using very powerful graph partitioning techniques to obtain a segmentation from the
input stacks.
The input of this widget should be the output of the [CNN-predictions widget](https://kreshuklab.github.io/plant-seg/chapters/plantseg_classic_gui/cnn_predictions/).
If the boundary prediction stage fails for any reason, a raw image could be used (especially if the cell boundaries are
 very sharp, and the noise is low) but usually does not yield satisfactory results.

* The **Algorithm** menu can be used to choose the segmentation algorithm. Available choices are:
    1. GASP (average): is a generalization of the classical hierarchical clustering. It usually delivers very
    reliable and accurate segmentation. It is the default in PlantSeg.
    2. MutexWS: Mutex Watershed is a derivative of the standard Watershed, where we do not need seeds for the
     segmentation. This algorithm performs very well in certain types of complex morphology (like )
    3. MultiCut: in contrast to the other algorithms is not based on a greedy agglomeration but tries to find the
    optimal global segmentation. This is, in practice, very hard, and it can be infeasible for huge stacks.
    4. DtWatershed: is our implementation of the distance transform Watershed. From the input, we extract a distance map
    from the boundaries. Based this distance map, seeds are placed at local minima. Then those seeds are used for
    computing the Watershed segmentation. To speed up the computation of GASP, MutexWS, and MultiCut, an over-segmentation
     is obtained using Dt Watershed.

* **Save Directory** defines the sub-directory's name where the segmentation results will be stored.

* The **Under/Over- segmentation factor** is the most critical parameters for tuning the segmentation of GASP,
MutexWS and MultiCut. A small value will steer the segmentation towards under-segmentation. While a high-value bias the
segmentation towards the over-segmentation. This parameter does not affect the distance transform Watershed.

* If **Run Watershed in 2D** value is True, the superpixels are created in 2D (over the z slice). While if False makes
the superpixels in the whole 3D volume. 3D superpixels are much slower and memory intensive but can improve
 the segmentation accuracy.

* The **CNN Predictions Threshold** is used for the superpixels extraction and Distance Transform Watershed. It has a
crucial role for the watershed seeds extraction and can be used similarly to the "Unde/Over segmentation factor."
to bias the final result.
A high value translates to less seeds being placed (more under segmentation), while with a low value, more seeds are
 placed (more over-segmentation).

* The input is used by the distance transform Watershed to extract the seed and find the segmentation boundaries.
If **Watershed Seeds Sigma** and **Watershed Boundary Sigma** are larger than
 zero, a gaussian smoothing is applied on the input before the operations above. This is mainly helpful for
 the seeds computation but, in most cases, does not impact segmentation quality.

* The **Superpixels Minimum Size** applies a size filter to the initial superpixels over-segmentation. This removes
Watershed often produces small segments and is usually helpful for the subsequent agglomeration.
 Segments smaller than the threshold will be merged with the nearest neighbor segment.

* Even though GASP, MutexWS, and MultiCut are not very prone to produce small segments, the **Cell Minimum Size** can
be used as a final size processing filter. Segments smaller than the threshold will be merged with the nearest
neighbor cell.
