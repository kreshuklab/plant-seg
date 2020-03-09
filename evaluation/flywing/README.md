## FlyWing evaluation

Running `flywing_eval.py` requires Python 2.7, so create new env before executing the script.

```bash
conda create -n fly-wing -c conda-forge h5py pillow futures python=2.7
conda activate flywing
pip install wget
```

### Evaluate
```bash
python flywing_eval.py --gt-dir GT_DIR --seg-dir SEG_DIR [--seg-dataset SEG_DATASET]
```
where:
* `GT_DIR` - directory containing ground truth files
* `SEG_DIR` - directory containing the segmentation files
* `SEG_DATASET` - segmentation inside the segmentation H5 file