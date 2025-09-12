# Batch processing

Plantseg supports processing a whole stack of images at once through the
use of batch scripts.

To generate such a workflow script, first process one image in the intended way,
and export the result. Then, go to the `Batch` tab in Napari and save the workflow
script to a `yaml` file.

You can review and edit workflow scripts using the build-in workflow editor;
open it either directly from the napari `Batch` tab, or from the command
line using `plantseg -e` .

Read more about [batch workflows in this chapter](../workflow_gui/index.md).

!!! note ""
    You find the batch processing section in the output tab in napari!

!!! warning
    Some interactive steps can't be recorded into workflows, especially **cropping** and **proofreading**

## Widget: Export Batch Workflow

```python exec="1" html="1"
--8<-- "widgets/batch/batch.py"
```
