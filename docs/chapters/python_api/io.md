# Input/Output

All the input/output operations are handled by the `plantseg.io` module. This module provides functions to read and write data in different formats. The supported formats are `tiff`, `h5`, and `zarr`, `jpeg`, `png`.

## Reading

::: plantseg.io.smart_load

## Writing

::: plantseg.io.create_tiff
::: plantseg.io.create_h5
::: plantseg.io.create_zarr

## Tiff Utilities

::: plantseg.io.tiff.read_tiff_voxel_size

## H5 Utilities

::: plantseg.io.h5.list_h5_keys
::: plantseg.io.h5.del_h5_key
::: plantseg.io.h5.rename_h5_key

## Zarr Utilities

::: plantseg.io.zarr.list_zarr_keys
::: plantseg.io.zarr.del_zarr_key
::: plantseg.io.zarr.rename_zarr_key
