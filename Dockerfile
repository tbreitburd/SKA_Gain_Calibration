FROM continuumio/miniconda3

RUN mkdir -p SKA_Coursework

COPY . /SKA_Coursework

WORKDIR /SKA_Coursework

RUN conda env update -f environment.yml --name SKA_CW

RUN apt-get update && apt-get install -y \
    git

RUN echo "conda activate SKA_CW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pip install pre-commit
RUN pre-commit install
