# Steps to run GIS supervised 3DGS for single object binary supervision

### The first thing is to have a GPU server. I am using a GPU server with CUDA 12.8.1 on Ubuntu 24.04 with PyTorch 2.13.0

### Please fork the repo in your workspace

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:anup39/gis-gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone git@github.com:anup39/gis-gaussian-splatting.git --recursive
```

### Download and Install Miniconda, `Miniconda3-latest-Linux-x86_64.sh`  using the following command 

``` shell 
# It shoud be in the path where you have download to execute the following command 
bash Miniconda3-latest-Linux-x86_64.sh 
/root/miniconda3/bin/conda --version
/root/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Install system dependencies 

``` shell
apt-get update
```

``` shell
apt-get install -y \
git wget ca-certificates \
build-essential ninja-build \
libgl1 libglib2.0-0 libegl1
```

### Create a conda env using 

```shell
## First accept terms and condtions 
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

## create environment
conda create -n gaussian_splatting python=3.11 pip -y
conda activate gaussian_splatting
python --version
which python

# Proceed to next step if the python version is correct
```

### Install now the python dependencies

```shell
python -m pip install --upgrade pip setuptools wheel ninja packaging
```

``` shell 
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```
### OR 

``` shell
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

``` shell 
pip install \
numpy==1.26.4 \
opencv-python-headless==4.11.0.86 \
open3d==0.19.0 \
plyfile==1.1 \
pillow tqdm scipy pyproj joblib tensorboard

```

### Apply a missing header fix 


``` shell 
nano /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
```
```
#include <cstdint>
#include <cstddef>
```

### Verify the selected CUDA environment 

``` shell 
python - <<'PY'
import os
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA_HOME:", os.environ["CUDA_HOME"])
print("Architecture:", os.environ["TORCH_CUDA_ARCH_LIST"])
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
PY
``` 

8. Apply the missing-header fix

### Complie the three extensions 

``` shell
pip install -v --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization
```
``` shell 
python -m pip install -v --no-build-isolation --no-deps ./submodules/simple-knn
```
```shell 
python -m pip install -v --no-build-isolation --no-deps ./submodules/fused-ssim
```

### Check if everything is running fine 

``` shell
python train.py --help
```
