## Conda Build

To manually create a conda package, run:

```bash
conda build -c conda-forge conda-recipe
```

## Release New Version on `conda-forge` Channel

1. Ensure `bump-my-version` is installed in your conda environment.
2. Checkout the `master` branch.
3. Run `bump-my-version bump patch` (or `major` or `minor`) to bump the version in `pyproject.toml` and `plantseg/__version__.py`, and create a new tag.
4. Push changes and tags to GitHub:

   ```bash
   git push && git push --tags
   ```

5. Create a new release on GitHub: [GitHub Releases](https://github.com/kreshuklab/plant-seg/releases).
6. Generate checksums for the new release:

   ```bash
   curl -sL https://github.com/kreshuklab/plant-seg/archive/refs/tags/VERSION.tar.gz | openssl sha256
   ```

   Replace `VERSION` with the new release version.

7. Fork the `conda-forge` feedstock repository: [conda-forge/plant-seg-feedstock](https://github.com/conda-forge/plant-seg-feedstock).
8. Clone the forked repository and create a new PR with the following changes:
    - Update the `version` in `recipe/meta.yaml` to the new release version.
    - Update the `sha256` in `recipe/meta.yaml` to the new checksum.

9. Wait for the checks to pass and ensure the PR is merged. Once the PR is merged, the new version will be available on the `conda-forge` channel.
