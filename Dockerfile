FROM nvidia/cuda:9.0-base-ubuntu16.04

MAINTAINER Scott Clay <scottclay8@gmail.com>

RUN apt-get update && \
    apt-get install -y curl build-essential libpng12-dev libffi-dev less vim  && \
    apt-get clean

RUN curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh && \
    bash /installer.sh -b -f && \
    rm /installer.sh

ENV PATH "$PATH:/root/anaconda3/bin"
ENV LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
ENV CUDA_HOME=/usr/local/cuda-9.0

ADD environment.yml /environment.yml
RUN conda update conda
RUN conda env update -f /environment.yml

RUN mkdir /root/.jupyter/
COPY jupyter_notebook_config.py /root/.jupyter/

RUN mkdir notebooks/
RUN mkdir scripts/
COPY scripts/train.py scripts/
COPY scripts/predict.py scripts/

COPY run_stuff.sh .
CMD ["./run_stuff.sh"]
