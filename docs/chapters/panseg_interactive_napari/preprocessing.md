# Preprocessing

This section describes the data processing functionalities available in
PanSeg Interactive. This functionalities are available in the
`Preprocessing` tab in the PanSeg Interactive GUI.

## Widget: Crop

<div>
<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="/path/to/poster.png">
    <source src="https://github.com/kreshuklab/panseg/raw/refs/heads/assets/videos/cropping_20fps.mp4" type="video/mp4">
  </video>
</figure>
</div>

To crop an image, first add a `Shapes` layer. You find the `Add Shapes Layer`
button above the layer list on the left side. With the `Shapes` layer selected,
you can add one rectangle using by pressing `R` and drawing.

In the Crop widget, you can also crop on the z dimension if applicable.

```python exec="1" html="1"
--8<-- "widgets/preprocessing/crop.py"
```

## Widget: Image Rescaling

Many boundary prediction models are sensitive to the scale of the input.
Use the `To model voxel size` option to match your input to your desired model.

--8<-- "widgets/preprocessing/rescale.md"

## Widget: Gaussian Smoothing

```python exec="1" html="1"
--8<-- "widgets/preprocessing/gaussian_smoothing.py"
```

## Widget: Image Pair Operations

```python exec="1" html="1"
--8<-- "widgets/preprocessing/image_pair_operations.py"
```
