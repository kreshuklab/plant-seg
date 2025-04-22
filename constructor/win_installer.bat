echo "### This is the main installation of PlantSeg ###"
CALL "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
tar xf build.gz
CALL conda install -y -c "%PREFIX%\conda_bld" -c conda-forge plantseg
echo 'Installation finished!'
