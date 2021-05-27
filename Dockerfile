FROM nvidia/cuda:10.1-base
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -n plant-seg -c pytorch -c conda-forge -c lcerrone -c abailoni -c cpape -c awolny pytorch nifty=1.0.9 plantseg

ENTRYPOINT ["conda", "run", "-n", "plant-seg", "plantseg", "--gui"]

