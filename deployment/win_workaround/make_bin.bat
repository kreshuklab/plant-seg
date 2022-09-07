@ECHO OFF
ECHO Creating GPU env
set version_gpu=plant-seg-gpu
echo y | call conda create -n %version_gpu% -c pytorch -c conda-forge -c lcerrone -c awolny python=3.9 pytorch-3dunet=1.3.7 plantseg
call conda pack -n %version_gpu% -o %version_gpu%-win64.zip

ECHO Creating CPU env
set version_cpu=plant-seg-cpu
echo y | call conda create -n %version_cpu% -c pytorch -c conda-forge -c lcerrone -c awolny python=3.9 pytorch-3dunet=1.3.7 plantseg cpuonly
call conda pack -n %version_cpu% -o %version_cpu%-win64.zip

ECHO Conda cleanup
echo y | call conda remove --name %version_gpu% --all
echo y | call conda remove --name %version_cpu% --all
call conda clean --all
