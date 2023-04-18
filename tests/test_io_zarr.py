"""
Testing io.zarr functionality.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University
"""


class TestZarr:

    def test_create_zarr(self, input_path_zarr, dataset_key="volumes/raw"):
        from plantseg.io.zarr import create_zarr
        import numpy as np
        stack_array = np.ones((32, 128, 128), dtype='float32')
        create_zarr(input_path_zarr, stack_array, dataset_key)

    def test_load_zarr(self, input_path_zarr, dataset_key="volumes/raw"):
        from plantseg.io.zarr import load_zarr, list_keys, create_zarr, rename_zarr_key
        # file load with specific dataset
        file, infos = load_zarr(path=input_path_zarr, key=dataset_key)
        print(f"file load with specific dataset:\n{file.shape} \n {infos}")

        # file load, with dataset key=None
        file, infos = load_zarr(path=input_path_zarr, key=None)
        print(f"file load with dataset key=None:\n{file} \n {infos}")

        # only info load
        infos = load_zarr(path=input_path_zarr, key=dataset_key, info_only=True)
        print(f"only info load:\n{infos}")

    def test_list_keys(self, input_path_zarr):
        from plantseg.io.zarr import list_keys
        print(list_keys(input_path_zarr))

    def test_rename_zarr_key(self, input_path_zarr, old_key="volumes/raw", new_key="volumes/raw2"):
        import zarr
        from plantseg.io.zarr import rename_zarr_key
        rename_zarr_key(input_path_zarr, old_key, new_key)

        # sanity check
        f = zarr.open(input_path_zarr, 'r')
        print(f.tree())

    def test_del_zarr_key(self, input_path_zarr, key="volumes/raw2"):
        import zarr
        from plantseg.io.zarr import del_zarr_key
        # sanity check
        f = zarr.open(input_path_zarr, 'r')

        print("## before deletion, checking hierarchy##")
        print(f.tree())

        # delete key
        del_zarr_key(input_path_zarr, key)

        print("## after deletion, checking hierarchy##")
        print(f.tree())

