# Additional Widgets

## Widget: Add Custom Model

```python exec="1" html="1"
--8<-- "napari/extra/widget_add_custom_model.py"
```

## Widget: Lifted Multi-Cut

As reported in our [paper](https://elifesciences.org/articles/57613), if one has a nuclei signal imaged together with
the boundary signal, we could leverage the fact that one cell contains only one nucleus and use the `LiftedMultict`
segmentation strategy and obtain improved segmentation.

```python exec="1" html="1"
--8<-- "napari/extra/widget_lifted_multicut.py"
```
