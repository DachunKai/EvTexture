# ========================================================================
# Module list for ``EvTexture: Event-driven Texture Enhancement for Video Super-resolution'' paper (ICML 2024).
# ------------------------------------------------------------------------
# python        3.7    (conda)
# pytorch       1.10.2+cu111    (pip)
# torchvision   0.11.3+cu111    (pip)
# BasicSR       1.4.2    (pip)
# ========================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
ENV PIP_ROOT_USER_ACTION=ignore

COPY torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl /tmp/
COPY torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl /tmp/
COPY miniconda.sh /tmp/
COPY ./EvTexture /EvTexture

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get upgrade -y && \
# ==================================================================
# environments
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        apt-utils \
        build-essential \
        ca-certificates \
        wget \
        cmake \
        git \
        g++ \
        gcc \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        && \
# =================================================================
# Miniconda3
# ----------------------------------------------------------------
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
# ================================================================
# dependencys for environment evtexture
    conda update -y conda && \
    conda create -y -n evtexture python=3.7 && \
    /bin/bash -c "source activate evtexture && pip install --upgrade pip && pip --no-cache-dir install /tmp/torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl && pip --no-cache-dir install /tmp/torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl && cd /EvTexture && pip --no-cache-dir install -r requirements.txt && python setup.py develop" && \
# =================================================================
# cleanup
#-----------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
