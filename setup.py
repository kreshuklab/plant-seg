from setuptools import setup, find_packages

exec(open('plantseg/__version__.py').read())
setup(
    name='plantseg',
    version=__version__,  # noqa: F821
    packages=find_packages(exclude=["tests", "evaluation"]),
    include_package_data=True,
    description='PlantSeg is a tool for cell instance aware segmentation in densely packed 3D volumetric images.',
    author='Lorenzo Cerrone, Adrian Wolny',
    url='https://github.com/kreshuklab/plant-seg',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)
