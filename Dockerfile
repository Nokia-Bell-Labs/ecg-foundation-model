FROM ubuntu:20.04

WORKDIR /

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y wget vim bzip2

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN rm /opt/conda/.condarc && \
	conda config --remove channels defaults && \
	conda config --add channels conda-forge

USER $USERNAME
