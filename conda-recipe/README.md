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

