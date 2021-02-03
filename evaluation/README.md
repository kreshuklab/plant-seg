# PlantSeg Evaluation Scripts
We provide with PlantSeg two main evaluation scripts. 
One for assessing the accuracy of the neural network and one for assessing segmentation accuracy.\
All files have to be converted to H5 to be evaluated with the evaluation scripts. 
The output evaluation will be saved on a standard CSV file.
## Segmentation Evaluation
The segmentation evaluation is like PlantSeg, config file based. 
To run the evaluation it is necessary to create in a text editor a YAML configuration file.
A complete [example](./evaluation_config.yml) is provided with comments for each field.\
Then the script can be run by executing in a terminal:
```bash
$ python evaluation_segmentation.py --config PATH_TO_CONFIG_FILE
```
The script must be executed from inside the evaluation directory.
## Neural Network Predictions Evaluation
The boundary predictions evaluation script can be run directly without any configuration file.
The script can be used in two main ways. 
* Single file: 
```bash
$ python evaluation_pmaps.py --gt PATH_TO_GROUNDTRUTH_FILE --predictions PATH_TO_PREDICTION_FILE
```
* Multiple files:
If the predictions are created with PlantSeg the naming should allow for automatic matching. 
Ground truth files and predictions must be the only H5 files inside their directory. 
```bash
$ python evaluation_pmaps.py --gt PATH_TO_GROUNDTRUTH_DIRECTORY --predictions PATH_TO_PREDICTION_DIRECTORY
```
Additional keys:
* threshold: Threshold at which the predictions will be binarized (float between 0-1).
* out-file: Define name (and location) of output file (final name: out-file + timestamp + .csv).
* p-key: Predictions dataset name inside the H5 file (default "predictions").
* gt-key: Ground truth dataset name inside the H5 file (default "label").
* sigma: Must match the default smoothing used in training. Default ovules 1.3.
