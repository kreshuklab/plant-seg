from setuptools import find_packages, setup

exec(open("panseg/__version__.py").read())
setup(
    name="panseg",
    version=__version__,  # noqa: F821
    packages=find_packages(exclude=["tests", "evaluation"]),
    include_package_data=True,
    package_data={
        "panseg": ["resources/logo_white.png"],
    },
    description="PanSeg is a tool for cell instance aware segmentation in densely packed 3D volumetric images.",
    author="Lorenzo Cerrone, Adrian Wolny, Qin Yu",
    url="https://github.com/kreshuklab/panseg",
    author_email="lorenzo.cerrone@uzh.ch, qin.yu@embl.de",
    entry_points={
        "console_scripts": [
            "panseg=panseg.run_panseg:main",
        ],
    },
)
