# Main PlantSeg Workflow

## Widget: Neural Network Prediction - Find Boundary

--8<-- "napari/main/widget_unet_prediction.md"

## Widget: Watershed - Generate Superpixels

```python exec="1" html="1"
--8<-- "napari/main/widget_dt_ws.py"
```

## Widget: Agglomeration - Merge Superpixels into Instances

```python exec="1" html="1"
--8<-- "napari/main/widget_agglomeration.py"
```
