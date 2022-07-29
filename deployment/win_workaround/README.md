# Deployment env requirements
* conda-pack
* pyinstaller
* pyyaml

## Create binary
Execute the following batch script in the Windows anaconda prompt to create the 
plantseg environment binaries.
```
cmd /c make_bin.bat
```

## Create exe
To create the `.exe` from the launcher script execute 
```
pyinstaller PlantSegLauncher.spec
```
in the anaconda prompt. The `.exe` will be placed inside `\dist`.