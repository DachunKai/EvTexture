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
        unzip \
        vim-gtk3 \
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
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
# ================================================================
# dependencys for environment evtexture
    conda update -y conda && \
    conda create -y -n evtexture python=3.7 && \
    /bin/bash -c "source activate evtexture && pip install --upgrade pip && pip --no-cache-dir install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html && cd /EvTexture && pip --no-cache-dir install -r requirements.txt && python setup.py develop" && \
# =================================================================
# cleanup
#-----------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
