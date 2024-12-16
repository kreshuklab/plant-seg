import numpy as np
from bioimageio.core.prediction import predict
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec.model.v0_5 import TensorId

array = np.random.randint(0, 255, (2, 128, 128, 128), dtype=np.uint8)
dims = ('c', 'z', 'y', 'x')
sample = Sample(members={TensorId('a'): Tensor(array=array, dims=dims)}, stat={}, id='try')

temp = predict(
    # model='https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/philosophical-panda/0.0.11/files/rdf.yaml',
    model='https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/emotional-cricket/1.1/files/rdf.yaml',
    # model='/Users/qin/Downloads/efficient-chipmunk.yaml',
    inputs=sample,
    sample_id='sample',
)
