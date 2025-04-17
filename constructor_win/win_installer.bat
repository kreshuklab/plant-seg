call "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
tar xf win_build.gz
echo "unzipped" >> log_installation
conda create -y -n plantseg python=3.12 menuinst
echo "created env" >> log_installation
conda activate plantseg
echo 'Activated, now installing Plantseg..' >> log_installation
conda install -v -c "%PREFIX%\conda_bld" -c conda-forge plantseg > log_installing
echo 'Installation finished!' >> log_installation
touch 'plantseg_installed'
conda info
conda list
sleep 20
