## Conda build

In order to create conda package manually:

Run:

```bash
conda build -c conda-forge conda-recipe1.8.3
```

## Release new version on `lcerrone` channel

1. Make sure that `bumpversion` is installed in your conda env
2. Checkout master branch
3. Run `bumpversion patch` (or `major` or `minor`) - this will bum the version in `.bumpversion.cfg` and `__version__.py` add create a new tag
4. Run `git push && git push --tags` - this will trigger tagged travis build
5. Tagged Travis build will do the following:
    - build a new conda package
    - deploy the new version of the conda package to anaconda cloud (`lcerrone` channel)

## Release new version on `conda-forge` channel

1. Make a new release on GitHub (https://github.com/kreshuklab/plant-seg/releases)
2. (Optional) Make sure that the new release version is in sync with the version in `.bumpversion.cfg` and `__version__.py` (see above)
3. Generate the checksums for the new release using: `curl -sL https://github.com/kreshuklab/plant-seg/archive/refs/tags/VERSION.tar.gz | openssl sha256`. Replace `VERSION` with the new release version
4. Fork the `conda-forge` feedstock  repository (https://github.com/conda-forge/plant-seg-feedstock)
5. Clone the forked repository and create a new PR with the following changes:
    - Update the `version` in `recipe/meta.yaml` to the new release version
    - Update the `sha256` in `recipe/meta.yaml` to the new checksum
6. Wait for the checks to pass. Make sure the PR is merged. Once the PR is merged, the new version will be available on the `conda-forge` channel
