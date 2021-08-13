### DeepDream-in-PyTorch

!(./.preview_img.jpg)


Run on Python 3.8 with PyTorch 1.9.0 and CUDA 11.1 (with cuDNN 8)

## Installation via pip
Install requirements inside your virtual environment as such
```bash
pip3 install -r requirements.txt
```
or like this, if the previous line did not work
```bash
cat requirements.txt | cut -f1 -d"#" | sed '/^\s*$/d' | xargs -n 1 pip install
```

## Docker
Can be run inside Docker container as such:
```bash
docker pull pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

docker run -it --rm --gpus all pytorch/pytorch bash

docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/" \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  anibali/pytorch bash

python3 train.py imgs/image_path.jpg
```
