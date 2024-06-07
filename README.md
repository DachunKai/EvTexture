# [EvTexture (ICML 2024)](https://icml.cc/virtual/2024/poster/34032)

<p align="left">
📃 <a href="https://drive.google.com/file/d/1RWptb35a-z-hwc3gZZY-FPd_G8g8Up1d/view?usp=sharing" target="_blank">[Paper]</a>
</p>

This is the official Pytorch implementation of "EvTexture: Event-driven Texture Enhancement for Video Super-Resolution" paper (ICML 2024).  This repository contains *video demos* and *codes* of our work.

**Authors**: [Dachun Kai](https://github.com/DachunKai/)<sup>[:email:️](mailto:dachunkai@mail.ustc.edu.cn)</sup>, Jiayao Lu, [Yueyi Zhang](https://scholar.google.com.hk/citations?user=LatWlFAAAAAJ&hl=zh-CN&oi=ao)<sup>[:email:️](mailto:zhyuey@ustc.edu.cn)</sup>, [Xiaoyan Sun](https://scholar.google.com/citations?user=VRG3dw4AAAAJ&hl=zh-CN), *University of Science and Technology of China*

**Feel free to ask questions. If you gain insights from our work, please don't hesitate to give us a :star:!**

## :rocket: News
- [ ] Release training code
- [ ] Release details to prepare datasets
- [ ] Publish docker image
- [x] Release pretrained models and test sets for quick testing
- [x] 2024/06/07: Video demos released
- [x] 2024/05/25: Initialize the repository
- [x] 2024/05/02: :tada: :tada: Our paper was accepted in ICML'2024

## :bookmark: Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Citation](#citation)
4. [License and Acknowledgement](#license-and-acknowledgement)
5. [Contact](#contact)

## :fire: Video Demos
A $4\times$ upsampling results on the [Vid4](https://paperswithcode.com/sota/video-super-resolution-on-vid4-4x-upscaling) and [REDS4](https://paperswithcode.com/dataset/reds) test sets. The videos have been compressed. Therefore, the results are inferior to that of the actual outputs.

https://github.com/DachunKai/EvTexture/assets/66354783/fcf48952-ea48-491c-a4fb-002bb2d04ad3

https://github.com/DachunKai/EvTexture/assets/66354783/ea3dd475-ba8f-411f-883d-385a5fdf7ff6

https://github.com/DachunKai/EvTexture/assets/66354783/e1e6b340-64b3-4d94-90ee-54f025f255fb

https://github.com/DachunKai/EvTexture/assets/66354783/01880c40-147b-4c02-8789-ced0c1bff9c4

## Code
### Model and results
Pre-trained models can be downloaded from [onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab), [google drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg) (n8hg).
* *EvTexture_REDS_BIx4.pth*: trained on REDS dataset with BI degradation for $4\times$ SR scale.
* *EvTexture_Vimeo90K_BIx4.pth*: trained on Vimeo-90K dataset with BI degradation for $4\times$ SR scale.

The output results on REDS4 and Vid4 can be downloaded from [onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab), [google drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing), and [baidu cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg) (n8hg).

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

## License and Acknowledgement
This project is released under the Apache-2.0 license. Our work is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), which is an open source toolbox for image/video restoration tasks. Thanks to the inspirations and codes from [RAFT](https://github.com/princeton-vl/RAFT) and [event_utils](https://github.com/TimoStoff/event_utils).


## Contact
If you meet any problems, please describe them in issues or contact:
* Dachun Kai: <dachunkai@mail.ustc.edu.cn>