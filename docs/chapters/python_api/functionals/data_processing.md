# Data Processing

Basic data processing functions are provided in the `dataprocessing` module. These functions are used to preprocess data before training a model, or to post-process the output of a model.

## Generic Functions

::: plantseg.functionals.dataprocessing.dataprocessing.normalize_01
::: plantseg.functionals.dataprocessing.dataprocessing.scale_image_to_voxelsize
::: plantseg.functionals.dataprocessing.dataprocessing.image_rescale
::: plantseg.functionals.dataprocessing.dataprocessing.image_median
::: plantseg.functionals.dataprocessing.dataprocessing.image_gaussian_smoothing
::: plantseg.functionals.dataprocessing.dataprocessing.image_crop
::: plantseg.functionals.dataprocessing.dataprocessing.process_images

## Segmentation Functions

::: plantseg.functionals.dataprocessing.labelprocessing.relabel_segmentation
::: plantseg.functionals.dataprocessing.labelprocessing.set_background_to_value

## Advanced Functions

::: plantseg.functionals.dataprocessing.advanced_dataprocessing.fix_over_under_segmentation_from_nuclei
::: plantseg.functionals.dataprocessing.advanced_dataprocessing.remove_false_positives_by_foreground_probability
