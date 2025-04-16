call "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
tar xf win_build.gz
if %errorlevel% neq 0 exit /b %errorlevel%

conda create -y -n plantseg python=3.12
conda activate plantseg
conda install -c conda-bld plantseg
