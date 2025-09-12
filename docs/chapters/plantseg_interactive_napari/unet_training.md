# UNet Training

With PlantSeg you can train bespoke segmentation models!
This is especially useful to get great results on a big dataset for which
the build-in models work not perfectly.  
First proofread some images from the dataset. Then train a new model on this
high-quality data, and run it on the whole dataset.

You can also fine-tune existing models on your data.

## Training from a dataset

For training from an dataset stored on disk, create the directories `train` and `val`.
Your training images must be stored as `h5` files in these directories.
The `h5` files must contain the input image under the `raw` key, and the
segmentation under the `label` key.

```
mydataset/
├── train/
│   ├── first.h5
│   └── second.h5
└── val/
    ├── val_one.h5
    └── val_two.h5 
```

## Train from GUI

To train from images loaded in the GUI, you need a layer containing the input
image and one containing the segmentation. Make sure the quality of the
segmentation is as good as possible by using the proofreading tool.

## Widgets

```python exec="1" html="1"
--8<-- "widgets/training/training.py"
```
