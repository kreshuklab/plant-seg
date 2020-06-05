# Segmentation
![alt text](./images/segmentation.png)

The segmentation widget allows to use very powerful graph partitioning techniques to obtain a segmentation from the
input stacks. 
The input of this widget should be the output of the [CNN-predictions widget](./Predictions.md). 
If the boundary prediction staged failed for any reason, a raw image can be used (especially if the cell boundaries are
 very sharp and the noise is low) but this usually does not yields satisfactory results.

* The **Algorithm** menu can be used to choose the segmentation algorithm, available choice are:
    1. GASP (average): is a generalization of the classical hierarchical clustering. It usually deliver very 
    reliable and accurate segmentation. It is the default in PlantSeg.
    2. MutexWS: Mutex Watershed is a derivative of the standard watershed where we do not need seeds for the
     segmentation. This algorithm performs very well in certain types of complex morphology (like )
    3. MultiCut: in contrast to the other algorithms is not based on a greedy agglomeration but tries to find the 
    global optimal segmentation. This is in practice very hard and it can infeasible for very large stacks.
    4. DtWateshed: is our implementation of the distance transform watershed. From the imput we extract a distance map 
    from the boundaries. Based this distance map, seeds are placed at local minima.  Then those seeds are used for 
    computing the watershed segmentation. To speed up the computation of GASP, MutexWS and MultiCut an over segmentation
     is obtained using Dt Watershed. 
   
* **Save Directory** defines the name of the sub-directory where the segmentation results will be stored.

* The **Under/Over- segmentation factor** is the most important parameters for tuning the segmentation of GASP,
MutexWS and MultiCut. A small value will steer the segmentation towards under-segmentation. While a high value bias the
segmentation towards the over-segmentation. This parameter has no effect for distance transform watershed.

* If **Run Watershed in 2D** value is True the superpixels are created in 2D (over the z slice). While if False creates
the superpixels in the whole 3D volume. 3D superpixels are much slower and memory intensive, but can improve
 the segmentation.
 
* The **CNN Predictions Threshold** is used for the superpixels extraction and Distance Transform Watershed. It has a 
crucial role for the watershed seeds extraction and can be used similarly to the "Unde/Over segmentation factor"
to bias the final result.  
An high value translate to less seeds being placed (more under segmentation),
 while with a low value more seeds are placed (more over segmentation).
   
* The input is used by the distance transform watershed for extracting the seed and 
 for finding the segmentation boundaries. If **Watershed Seeds Sigma** and **Watershed Boundary Sigma** are larger than
 zero a gaussian smoothing is applied on the input before the aforementioned operations. This is mainly helpful for
 the seeds computation, but in most cases does not impact segmentation quality.
 
* The **Superpixels Minimum Size** applies a size filter to the initial superpixels over segmentation, this remove 
small segments that are often produced by watershed. This is usually helpful for the subsequent agglomeration.
 Segments smaller than the threshold will be merged with a the nearest neighbour segment. 
 
* Even tough GASP, MutexWS and MultiCut are not very prone to produce small segments, the **Cell Minimum Size** can 
be uses a final size processing filter. Segments smaller than the threshold will be merged with a the nearest 
neighbour cell. 