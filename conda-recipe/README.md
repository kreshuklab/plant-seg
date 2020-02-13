## Conda build
In order to create conda package:

Run: 
```
conda build conda-recipe -c abailoni -c cpape -c awolny -c conda-forge
```

## Docker image
In order to create docker image:
```
sudo docker build -t plantseg .
sudo docker tag plantseg wolny/plantseg:<version>
sudo docker tag plantseg wolny/plantseg:latest
```
Push the image to Docker Hub:
```
sudo docker push wolny/plantseg:<version>
sudo docker push wolny/plantseg:latest
```

## Release new version
1. Make sure that `bumpversion` is installed in your conda env
2. Checkout master branch
3. Run `bumpversion patch` (or `major` or `minor`) - this will bum the version in `.bumpversion.cfg` and `__version__.py` add create a new tag.
4. Run `git push --follow-tags` - this will trigger tagged travis build
5. Travis will build the conda package and deploy the new version to anaconda cloud. Make sure you have the `CONDA_UPLOAD_TOKEN` setup in Travis.