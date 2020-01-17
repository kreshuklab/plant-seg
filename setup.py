from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='plantseg',
    version=__version__,
    packages=find_packages(exclude=["tests", "evaluation"]),
    include_package_data=True,
    description='PlantSeg is a tool for cell instance aware segmentation in densely packed 3D volumetric images.',
    author='Lorenzo Cerrone, Adrian Wonly',
    url='https://github.com/hci-unihd/plant-seg',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)