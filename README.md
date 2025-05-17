## :bookmark: Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Citation](#citation)


## :fire: Video Demos

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
    cd EvTexture && pip install -r requirements.txt && python setup.py develop
    ```
* Run in Docker :clap:

  Note: before running the Docker image, make sure to install nvidia-docker by following the [official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

  [Option 1] Directly pull the published Docker image we have provided from [Alibaba Cloud](https://cr.console.aliyun.com/cn-hangzhou/instances).
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/dachunkai/evtexture:latest
  ```

  [Option 2] We also provide a [Dockerfile](https://github.com/DachunKai/EvTexture/blob/main/docker/Dockerfile) that you can use to build the image yourself.
  ```bash
  cd EvTexture && docker build -t evtexture ./docker
  ```
  The pulled or self-built Docker image containes a complete conda environment named `evtexture`. After running the image, you can mount your data and operate within this environment.
  ```bash
  source activate evtexture && cd EvTexture && python setup.py develop
  ```
### Test
1. Download the pretrained models from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)) and place them to `experiments/pretrained_models/EvTexture/`. The network architecture code is in [evtexture_arch.py](https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py).
    * *EvTexture_REDS_BIx4.pth*: trained on REDS dataset with BI degradation for $4\times$ SR scale.
    * *EvTexture_Vimeo90K_BIx4.pth*: trained on Vimeo-90K dataset with BI degradation for $4\times$ SR scale.

2. Download the preprocessed test sets (including events) for REDS4 and Vid4 from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)), and place them to `datasets/`.
    * *Vid4_h5*: HDF5 files containing preprocessed test datasets for Vid4.

    * *REDS4_h5*: HDF5 files containing preprocessed test datasets for REDS4.

3. Run the following command:
    * Test on Vid4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_Vid4_BIx4.yml
      ```
    * Test on REDS4 for 4x VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_REDS4_BIx4.yml
      ```
      This will generate the inference results in `results/`. The output results on REDS4 and Vid4 can be downloaded from ([Releases](https://github.com/DachunKai/EvTexture/releases) / [Onedrive](https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab) / [Google Drive](https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing) / [Baidu Cloud](https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg)(n8hg)).

### Data Preparation
* Both video and event data are required as input, as shown in the [snippet](https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py#L70). We package each video and its event data into an [HDF5](https://docs.h5py.org/en/stable/quick.html#quick) file.

* Example: The structure of `calendar.h5` file from the Vid4 dataset is shown below.

  ```arduino
  calendar.h5
  ├── images
  │   ├── 000000 # frame, ndarray, [H, W, C]
  │   ├── ...
  ├── voxels_f
  │   ├── 000000 # forward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ├── voxels_b
  │   ├── 000000 # backward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ```
* To simulate and generate the event voxels, refer to the dataset preparation details in [DataPreparation.md](https://github.com/DachunKai/EvTexture/blob/main/datasets/DataPreparation.md).

### Inference on your own video
:hammer_and_wrench: We are developing a convenient script to allow users to quickly use our EvTexture model to upscale their own videos. However, our spare time is limited, so please stay tuned!

## 😊 Citation and References

This project is inspired by and builds upon the following works:

- **EvTexture: Event-driven Texture Enhancement for Video Super-Resolution**  
  Project page: [https://dachunkai.github.io/evtexture.github.io/](https://dachunkai.github.io/evtexture.github.io/)  
  arXiv preprint: [https://arxiv.org/abs/2406.13457](https://arxiv.org/abs/2406.13457)

If you use this code or ideas in your work, please also consider citing the original paper:

```bibtex
@inproceedings{kai2024evtexture,
  title={{E}v{T}exture: {E}vent-driven {T}exture {E}nhancement for {V}ideo {S}uper-{R}esolution},
  author={Kai, Dachun and Lu, Jiayao and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={22817--22839},
  year={2024},
  volume={235},
  publisher={PMLR}
}
