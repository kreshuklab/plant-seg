# Contribute to PlantSeg

PlantSeg is an open-source project, and we welcome contributions from the community. There are many ways to contribute, such as writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests, or writing code that can be incorporated into PlantSeg itself.

## Getting Started

To set up the development environment, run:

```bash
mamba env create -f environment-dev.yml
conda activate plant-seg-dev
```

To install PlantSeg in development mode, run:

```bash
pip install -e . --no-deps
```

## Coding Style

PlantSeg uses _Ruff_ for linting and formatting. _Ruff_ is compatible with _Black_ for formatting. Ensure you have _Black_ set as the formatter with a line length of 120.

## Before Submitting a Pull Request

### Run Tests with `pytest`

Ensure that `pytest` is installed in your conda environment. To run the tests, simply use:

```bash
pytest
```

### Check Syntax with `pre-commit`

The PlantSeg repository uses pre-commit hooks to ensure the code is correctly formatted and free of linting issues. While not mandatory, it is encouraged to check your code before committing by running:

```bash
pre-commit run --all-files
```

For efficiency, pytest is not included in the pre-commit hooks. Please run the tests separately.
