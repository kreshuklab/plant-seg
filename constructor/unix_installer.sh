echo "### This is the main installation of PlantSeg ###"
. "${PREFIX}/etc/profile.d/conda.sh" && conda activate "${PREFIX}"
cd "${PREFIX}"
tar xf build.gz
conda install -y -c "${PREFIX}/conda_bld" -c conda-forge plantseg
