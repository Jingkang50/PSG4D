# 4D Panoptic Scene Graph Generation
<p align="center">

| ![pvsg.jpg](assets/teaser.png) |
|:--:|

  <p align="center">
  <a href="https://arxiv.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-NeurIPS%202023-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpHpnXP-ta9Nu1wD6FwkDWAB0LxY8oE9VNqsgv6ln-i8QQ?e=fURefF" target='_blank'>
    <img src="https://img.shields.io/badge/Data-PSG4D-334b7f?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgvpTfCTMudLpxw-h0_BVdcBAHacUaAQD-u9OvkUlpaDBg?e=LXnqaX" target='_blank'>
    <img src="https://img.shields.io/badge/Data-QuickView-7de5f6?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/jingkang50/PSG4D" target='_blank'>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fjingkang50%2FPSG4D&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=true">
  </p>
  </a>
  <p align="center">
  <font size=5><strong>4D Panoptic Scene Graph Generation</strong></font>
    <br>
        <a href="https://jingkang50.github.io/">Jingkang Yang</a>,
        <a href="https://cen-jun.com/">Jun Cen</a>,
        <a href="https://lilydaytoy.github.io/">Wenxuan Peng</a>,
        <a href="https://github.com/choiszt">Shuai Liu</a>,<br>
        <a href="https://hongfz16.github.io/=">Fangzhou Hong</a>,
        <a href="https://lxtgh.github.io/">Xiangtai Li</a>,
        <a href="https://kaiyangzhou.github.io/">Kaiyang Zhou</a>,
        <a href="https://cqf.io/">Qifeng Chen</a>,
        <a href="https://liuziwei7.github.io/">Ziwei Liu</a>,
    <br>
  S-Lab, NTU & HKUST & BUPT & HKBU
  </p>
</p>

---
## What is PSG4D Task?
<strong>The PSG4D (4D Panoptic Scene Graph Generation) Task</strong> is a novel task that aims to bridge the gap between raw visual inputs in a dynamic 4D world and high-level visual understanding. It involves generating a comprehensive 4D scene graph from RGB-D video sequences or point cloud video sequences.

## The PSG4D Dataset

We provide two dataset to facilitate PSG4D research. To access them, please checkout `data/GTA` and `data/HOI`.

<div style="display: flex; justify-content: space-around;">
  <div style="width: 45%;">
    <iframe src="https://player.vimeo.com/video/947191752?h=b912b9e0b6" width="100%" height="300" frameborder="0" allowfullscreen></iframe>
    <p style="text-align: center;">PSG4D-GTA Dataset Demo</p>
  </div>
  <div style="width: 45%;">
    <iframe src="https://player.vimeo.com/video/947193001?h=33c580b86a" width="100%" height="300" frameborder="0" allowfullscreen></iframe>
    <p style="text-align: center;">PSG4D-HOI Dataset Demo</p>
  </div>
</div>

## Get Started



## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{yang2023psg4d,
    author = {Yang, Jingkang and Cen, Jun and Peng, Wenxuan and Liu, Shuai amd Hong, Fangzhou and Li, Xiangtai and Zhou, Kaiyang and Chen, Qifeng and Liu, Ziwei}
    title = {4D Panoptic Scene Graph Generation},
    booktitle = {NeurIPS},
    year = {2023},
}
```