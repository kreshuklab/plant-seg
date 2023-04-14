"""
Testing io.zarr functionality.

Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University
"""

import queue

import pytest

from plantseg.io.zarr import *


def test_load_zarr(path, dataset_key):
    # file load with specific dataset
    file, infos = load_zarr(path=path, key=dataset_key)
    print(f"file load with specific dataset:\n{file} \n {infos}")

    # file load, with dataset key=None
    file, infos = load_zarr(path=path, key=None)
    print(f"file load with dataset key=None:\n{file} \n {infos}")

    # only info load
    infos = load_zarr(path=path, key=dataset_key, info_only=True)
    print(f"only info load:\n{infos}")


def test_list_keys(path):
    print(list_keys(path))


def test_create_zarr(path, dataset_key):
    stack_array = np.ones((5, 500, 600), dtype='float32')
    create_zarr(path, stack_array, dataset_key)


def test_rename_zarr_key(path, old_key, new_key):
    rename_zarr_key(path, old_key, new_key)

    # sanity check
    f = zarr.open(path, 'r')
    print(f.tree())


def test_del_zarr_key(path, key):
    # sanity check
    f = zarr.open(path, 'r')

    print("## before deletion, checking hierarchy##")
    print(f.tree())

    # delete key
    del_zarr_key(path, key)

    print("## after deletion, checking hierarchy##")
    print(f.tree())


def main():
    # path to local zarr and dataset in it
    path = "/home/samia/Documents/test-plantseg/data/new_filename.zarr"

    dataset_key = "volumes/raw"

    # create zarr
    test_create_zarr(path, dataset_key)

    # list keys
    test_list_keys(path)

    # load, _find_input_key, read_zarr_voxel_size
    test_load_zarr(path, dataset_key)

    # rename dataset key/name in the zarr file
    test_rename_zarr_key(path, dataset_key, new_key='volumes/raw_2')

    # delete dataset key/name in the zarr file
    test_del_zarr_key(path,  "volumes/raw_2")


if __name__ == '__main__':
    main()
