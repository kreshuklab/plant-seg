call "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
set CONDA_OVERRIDE_CUDA="12.4"
tar xf win_build.gz
if %errorlevel% neq 0 exit /b %errorlevel%

conda create -y -n plantseg python=3.12 menuinst
conda activate plantseg
echo 'Activated, now installing Plantseg..'
conda install -v -c "%PREFIX%\conda_bld" -c conda-forge plantseg
echo 'Installation finished!'
touch 'plantseg_installed'
conda info
conda list
sleep 20
