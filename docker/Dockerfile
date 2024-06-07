# ========================================================================
# Module list for ``EvTexture: Event-driven Texture Enhancement for Video Super-resolution'' paper (ICML 2024).
# ------------------------------------------------------------------------
# python        3.7    (conda)
# pytorch       1.10.2+cu111    (pip)
# torchvision   0.11.3+cu111    (pip)
# BasicSR       1.4.2    (pip)
# ========================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

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
# Torch, Torchvision and EvTexture
# ----------------------------------------------------------------
    wget --quiet https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl -O /tmp/torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl && \
    wget --quiet https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl -O /tmp/torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl && \
    git clone https://github.com/DachunKai/EvTexture -O /EvTexture/ && \
# =================================================================
# Miniconda3
# ----------------------------------------------------------------
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
# ================================================================
# dependencys for environment evtexture
    conda update conda && \
    conda create -y -n evtexture python=3.7 && \
    /bin/bash -c "source activate evtexture && pip install --upgrade pip && pip --no-cache-dir install /tmp/torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl && pip --no-cache-dir install /tmp/torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl && cd /EvTexture && pip --no-cache-dir install -r requirements.txt && python setup.py develop" && \
# =================================================================
# cleanup
#-----------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*