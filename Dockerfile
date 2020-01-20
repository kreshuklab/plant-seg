FROM continuumio/miniconda3
RUN conda create -n plant-seg -c lcerrone -c abailoni -c cpape -c awolny -c conda-forge plantseg
ENTRYPOINT ["conda", "run", "-n", "plant-seg", "plantseg", "--gui"]
