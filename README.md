# [EvTexture (ICML 2024)](https://icml.cc/virtual/2024/poster/34032)

<p align="left">
📃 <a href="https://drive.google.com/file/d/1RWptb35a-z-hwc3gZZY-FPd_G8g8Up1d/view?usp=sharing" target="_blank">[Paper]</a>
</p>

This is the official Pytorch implementation of "EvTexture: Event-driven Texture Enhancement for Video Super-Resolution" paper (ICML 2024).  This repository contains *video demos* and *codes* of our work.

**Authors**: [Dachun Kai](https://github.com/DachunKai/)<sup>[:email:️](mailto:dachunkai@mail.ustc.edu.cn)</sup>, Jiayao Lu, [Yueyi Zhang](https://scholar.google.com.hk/citations?user=LatWlFAAAAAJ&hl=zh-CN&oi=ao)<sup>[:email:️](mailto:zhyuey@ustc.edu.cn)</sup>, [Xiaoyan Sun](https://scholar.google.com/citations?user=VRG3dw4AAAAJ&hl=zh-CN), *University of Science and Technology of China*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**

## :rocket: News
- [ ] Release training code
- [ ] Release details to prepare datasets
- [x] 2024/06/08: Publish docker image
- [x] 2024/06/08: Release pretrained models and test sets for quick testing
- [x] 2024/06/07: Video demos released
- [x] 2024/05/25: Initialize the repository
- [x] 2024/05/02: :tada: :tada: Our paper was accepted in ICML'2024

## :bookmark: Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Citation](#citation)
4. [Contact](#contact)
5. [License and Acknowledgement](#license-and-acknowledgement)

## :fire: Video Demos
A $4\times$ upsampling results on the [Vid4](https://paperswithcode.com/sota/video-super-resolution-on-vid4-4x-upscaling) and [REDS4](https://paperswithcode.com/dataset/reds) test sets.

https://github.com/DachunKai/EvTexture/assets/66354783/fcf48952-ea48-491c-a4fb-002bb2d04ad3

https://github.com/DachunKai/EvTexture/assets/66354783/ea3dd475-ba8f-411f-883d-385a5fdf7ff6

https://github.com/DachunKai/EvTexture/assets/66354783/e1e6b340-64b3-4d94-90ee-54f025f255fb

https://github.com/DachunKai/EvTexture/assets/66354783/01880c40-147b-4c02-8789-ced0c1bff9c4

## Code
### Installation
* Dependencies: [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), [CUDA Toolkit 11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive), [torch 1.10.2+cu111](https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl), and [torchvision 0.11.3+cu111](https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl).

* Run in Conda

    ```bash
    conda create -y -n evtexture python=3.7
    conda activate evtexture
    pip install torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl
    pip install torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl
    git clone https://github.com/DachunKai/EvTexture.git
    cd /path/to/EvTexture
    pip install -r requirements.txt
    python setup.py develop
    ```
* Run in Docker :clap:

  Note: before running the Docker image, make sure to install nvidia-docker by following the [official intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

  [Option 1] Directly pull the published Docker image we have provided from [Dockerhub](https://hub.docker.com/)/[Alibaba Cloud](https://cr.console.aliyun.com/cn-hangzhou/instances).
  ```bash
  docker pull dachunkai/evtexture:latest # From Dockerhub
  # or
  docker pull registry.cn-hangzhou.aliyuncs.com/dachunkai/evtexture:latest # From Alibaba Cloud
  ```

  [Option 2] We also provide a [Dockerfile](docker/Dockerfile) that you can use to build the image yourself.
  ```bash
  cd EvTexture && docker build -t evtexture ./docker
  ```
  The pulled or self-built Docker image containes a complete conda environment named `evtexture`. After running the image, you can mount your data and operate within this environment.
  ```bash
  source activate evtexture && cd EvTexture && python setup.py develop
  ```
### Test
1. Download the pretrained models to `experiments/pretrained_models/EvTexture/`. ([Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab)/[Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing)/[Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)). The network architecture code is in [evtexture_arch.py](https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py).
    * *EvTexture_REDS_BIx4.pth*: trained on REDS dataset with BI degradation for $4\times$ SR scale.
    * *EvTexture_Vimeo90K_BIx4.pth*: trained on Vimeo-90K dataset with BI degradation for $4\times$ SR scale.

2. Download the preprocessed test sets (including events) for REDS4 and Vid4 to `datasets/`. ([Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab)/[Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing)/[Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg))
    * *Vid4_h5*: HDF5 files containing preprocessed test datasets for Vid4.

    * *REDS4_h5*: HDF5 files containing preprocessed test datasets for REDS4.

3. Run the following command:
    * Test on Vid4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh 1 options/test/EvTexture/test_EvTexture_Vid4_BIx4.yml
      ```
    * Test on REDS4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh 1 options/test/EvTexture/test_EvTexture_REDS4_BIx4.yml
      ```
      This will generate the inference results in `results/`.

4. The output results on REDS4 and Vid4 can be downloaded from ([Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab)/[Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing)/[Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)). Each inference frame is named `f"{frame_index:06d}_{PSNR:.4f}_EvTexture_{dataset}_BIx4.png"`.



## :blush: Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :smiley:
```
@inproceedings{kai2024evtexture,
  title={Ev{T}exture: {E}vent-driven {T}exture {E}nhancement for {V}ideo {S}uper-{R}esolution},
  author={Kai, Dachun and Lu, Jiayao and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}

```

## Contact
If you meet any problems, please describe them in issues or contact:
* Dachun Kai: <dachunkai@mail.ustc.edu.cn>

## License and Acknowledgement
This project is released under the Apache-2.0 license. Our work is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), which is an open source toolbox for image/video restoration tasks. Thanks to the inspirations and codes from [RAFT](https://github.com/princeton-vl/RAFT) and [event_utils](https://github.com/TimoStoff/event_utils).
