# Deep Hierarchical Semantic Segmentation

This repo contains the unoffical supported code and configuration files to reproduce semantic segmentaion results of [HieraSeg](https://arxiv.org/abs/2203.14335). It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation).


## Usage

### Installation 

```
conda create -n hssn python==3.8
conda activate hssn
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
mim install mmcv-full
pip install mmsegmentation
pip install mmcv==1.6.1
```

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation.


### Structure
Follow this structure:
```
|-- datasets
        |-- Pascal-Person-Part
             |-- train
             |-- validation
|-- DH_HSSN
        |-- iter_6000-74.13.pth
```



### Inference
```
# single-gpu testing
python tools/test.py configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_60k_pascal_person_part_hiera_triplet.py /mnt/server14_hard0/msson/HSSN_pytorch/output_iter60k+-res101/iter_6000-74.13.pth --eval mIoU --show-dir ./visualization

# multi-gpu, multi-scale testing
tools/dist_test.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_60k_pascal_person_part_hiera_triplet.py /mnt/server14_hard0/msson/HSSN_pytorch/output_iter60k+-res101/iter_6000-74.13.pth 4 --aug-test --eval mIoU
```


## Results and Models

| Dataset | Backbone | Crop Size | mIoU (single scale) | mIoU (multi scale w/ flipping) | config |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Pascal-Person-Part | ResNet101 | 480x480 | 73.44 | 75.4 | [config](configs/deeplabv3plus/deeplabv3plus_r101-d8_480x480_60k_pascal_person_part_hiera_triplet.py) |



single scale             |  multi scale w/ flipping
:-------------------------:|:-------------------------:
![single](https://user-images.githubusercontent.com/66418958/205012632-647389ca-8aca-41f7-9b9c-6f4236dc0462.png)  |  ![multi](https://user-images.githubusercontent.com/66418958/205012141-4713416d-b125-4145-9111-de41e25cd00c.png)



## Citing HieraSeg
```BibTeX
@article{li2022deep,
  title={Deep Hierarchical Semantic Segmentation},
  author={Li, Liulei and Zhou, Tianfei and Wang, Wenguan and Li, Jianwu and Yang, Yi},
  journal={CVPR},
  year={2022}
}
```
