=== "From factor"
    Using the `From factor` mode, the user can rescale the image by a multiplicate factor.
    For example, if the image has a shape `(10, 10, 10)` and the user wants to rescale it by a factor of `(2, 2, 2)`, the new size will be `(20, 20, 20)`.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_FROM_FACTOR.py"
    ```

=== "To layer voxel size"
    Using the `To layer voxel size` mode, the user can rescale the image to the voxel size of a specific layer.
    For example, if two images are loaded in the viewer, one with a voxel size of `(0.1, 0.1, 0.1)um` and the other with a voxel size of `(0.1, 0.05, 0.05)um`, the user can rescale the first image to the voxel size of the second image.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_TO_LAYER_VOXEL_SIZE.py"
    ```

=== "To layer shape"
    Using the `To layer shape` mode, the user can rescale the image to the shape of a specific layer. For example, if two images are loaded in the viewer, one with a shape `(10, 10, 10)` and the other with a shape `(20, 20, 20)`, the user can rescale the first image to the shape of the second image.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_TO_LAYER_SHAPE.py"
    ```

===+ "To model voxel size"
    Using the `To model voxel size` mode, the user can rescale the image to the voxel size of the model.
    For example, if the model has been trained with data at voxel size of `(0.1, 0.1, 0.1)um`, the user can rescale the image to this voxel size.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_TO_MODEL_VOXEL_SIZE.py"
    ```

=== "To voxel size"
    Using the `To voxel size` mode, the user can rescale the image to a specific voxel size.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_TO_VOXEL_SIZE.py"
    ```

=== "To shape"
    Using the `To shape` mode, the user can rescale the image to a specific shape.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_TO_SHAPE.py"
    ```

=== "Set voxel size"
    Using the `Set voxel size` mode, the user can set the voxel size of the image to a specific value. This only changes the metadata of the image and does not rescale the image.

    ```python exec="1" html="1"
    --8<-- "widgets/preprocessing/rescale_SET_VOXEL_SIZE.py"
    ```
