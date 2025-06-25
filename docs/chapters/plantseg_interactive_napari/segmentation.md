# Segmentation

The segmentation workflow consists of three main steps:

- Boundary Prediction
- Boundary to Superpixels
- Superpixels to Segmentation

## Widget: 1. Boundary Predictions

Choose one of the build-in PlantSeg models, or one from the BioImage.IO Model Zoo.

Alternatively, you can import your own model by choosing `ADD CUSTOM MODEL`
from the model selection drop-down menu.

--8<-- "widgets/segmentation/prediction.md"

## Widget: 2. Boundary to Superpixels

Here, the boundary prediction is turned into superpixels by using
distance transform watershed.

```python exec="1" html="1"
--8<-- "widgets/segmentation/watershed.py"
```

## Widget: 3. Superpixels to Segmentation

DT Watershed tends to over-segment the image, therefor an agglomeration algorithm
is used in this third step.

```python exec="1" html="1"
--8<-- "widgets/segmentation/agglomeration.py"
```
