# Manual Proofreading

<div>
<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="/path/to/poster.png">
    <source src="https://github.com/kreshuklab/plant-seg/raw/refs/heads/assets/videos/proofreading_20fps.mp4" type="video/mp4">
  </video>
</figure>
</div>

## Widget: Proofreading with Manual Split and Merge

Once the segmentation is done, or a label image is loaded along with a boundary image, the user can start the proofreading process. In the proofreading widget, set the "Mode" to "Layer" if you are going to correct a loaded image, or set the "Mode" to "File" if you already had a proofreading session exported as a file. For "Segmentation", select the image layer that you want to correct, then click "Initialise" to start the proofreading process.

After a few second of initialisation, the initialised proofreading widget will be displayed, and two new layers will be added to the Napari viewer, "Scribbles" and "Correct Labels". Click "Scribbles" first, and you may draw on this layer with the "paint brush" tool - you can press `2` or `P` key to activate the brush tool or find it on the "layer controls" panel of the Napari viewer. The scribbles drawn on this layer will be used to split or merge the segmentation: strokes with the same color will merge the instance(s) in the segmentation layer, and strokes with different colors will split the instance(s) in the segmentation layer. After drawing the scribbles, click "Split / Merge" or press `n` to apply the scribbles to the segmentation layer. If you want to undo the last split or merge operation, click "Undo Last Action".

When you are done or want to stop the proofreading session, you may save the current state of the proofreading session by clicking "Save current proofreading snapshot" into a snapshot file, which can be loaded into the proofreading widget later in another session of PlantSeg.

### Keybinding

* `s`: Save stack
* `n`: Merge or split from seeds
* `ctrl+n`: Undo merge or split from seeds
* `c`: Clean seeds
* `o`: Mark/un-mark correct segmentation
* `b`: show/un-show correct segmentation layer
* `j`: Update boundaries from segmentation
* `k`: Update segmentation from boundaries
* `ctrl + arrows`: to move the field of view
* `alt + down/up arrows`: to increase or decrease the field of view
