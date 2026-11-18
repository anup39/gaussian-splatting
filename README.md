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

### If miniconda pacakges missing repo __init__ contain then please run following without this you will face some import error:

``` shell 
cp /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py \
   /root/miniconda3/envs/gaussian_splatting/lib/python3.11/site-packages/diff_gaussian_rasterization/__init__.py
```

### Environment variables 

#### This is where you SFM undistorted directory 
``` shell
export DATA=/workspace/OpenSfM/data/uav/undistorted 
```
#### This is where you images are that are undistorted (removal of radial distortions)
```shell
export IMAGES=$DATA/images
```
#### The SFM reconstruction json in local coordinates 
```shell
export RECON=/workspace/OpenSfM/data/uav/reconstruction.json
```
#### The referece lat, lon and alt of the reconstruction
```shell
export REFERENCE_LLA=/workspace/OpenSfM/data/uav/reference_lla.json
```
#### Camera params, example  should be pinhole model, here the radial distortions are removed
```
{
    "width": 5280,
    "height": 3956,
    "focal_x": 3725.151611,
    "focal_y": 3725.151611,
    "c_x": 2640.0,
    "c_y": 1978.0,
    "k_1": 0.0,
    "k_2": 0.0
}
```
```shell
export CAMERA_PARAMS=$DATA/camera_params.json
```

##### The geojson of the footprint that needs to be used for supervision
```shell
export GEOJSON=$DATA/stockpile.geojson
```
#### This is where you want to place you masking related output
```shell
export MASK_ROOT=$DATA/semantic_mask_folder
```
#### The acutal local where masks are stored
```shell
export MASK_DIR=$MASK_ROOT/mask
```
#### This contains debug images for masks for manual verification
```shell
export DEBUG_DIR=$MASK_ROOT/masks_debug
```
#### This is where the evaluation masks are which are annoted (I used LabelMe but you can you other software as well)
```shell
export MANUAL_MASKS=$DATA/evaluation_masks_manual
```
#### The  where you want to place you 3DGS outputs 
```shell
export OUT=/workspace/gaussian-splatting/output/gis_supervised
```
#### Location for base supervision
```shell
export B0=$OUT/B0_rgb
```
#### Location for gis supervision without boundary loss
```shell
export A0=$OUT/A0_gis_semantic
```
#### Location for gis supervision with boundary loss
```shell
export A0_B=$OUT/A0_B_gis_semantic
```
#### Incase you have not created the working directories 
``` shell
mkdir -p "$MASK_DIR" "$DEBUG_DIR" "$MANUAL_MASKS" "$OUT" 
```


## Exmaple 3 : Lets run now the GIS supervised with boundary loss

``` shell 
python train.py \
  -s "$DATA" \
  -m "$A0_B" \
  -r 4 \
  --eval \
  --eval_split_file "$DATA/eval_holdout.txt" \
  --iterations 30000 \
  --densify_grad_threshold 0.0004 \
  --densify_until_iter 10000 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --seed 42 \
  --disable_viewer \
  --gis_mask_dir "$MASK_DIR" \
  --geo_mask_weight 0.02 \
  --geo_normalize_semantic \
  --geo_start_iteration 5000 \
  --geo_ramp_iterations 5000 \
  --geo_geometry_gradient_scale 0.0 \
  --geo_boundary_ignore_px 8 \
  --geo_boundary_min_weight 0.05 \
  --geo_min_coverage 0.01 \
  --geo_boundary_weight 0.03 \
  --geo_boundary_band_px 8 \
  --geo_densify_weight 0.0 \
  --geo_cross_view_weight 0.0
```

## Evaluation : 

### Render held out RGB Images
``` shell
for MODEL in \
  $A0_B
do
  python render.py \
    -m $MODEL \
    --iteration 30000 \
    --skip_train
done
```

### Calculate PSNR, SSIM and LPIPS
```shell
python metrics.py \
  -m \
  $A0_B
```

### Render Semantic probability maps 

```shell 
python gis/render_gis_semantics.py \
  -m $A0_B \
  --iteration 30000 \
  --skip_train \
  --minimum_coverage 0.01
``` 

### For validation datasets evaluate 

``` shell
python gis/evaluate_gis_semantics.py \
  --prediction-dir "$A0_B/test/ours_30000/gis_semantics/probability_u16" \
  --reference-dir "$MANUAL_MASKS" \
  --include-file "$DATA/eval_validation.txt" \
  --threshold 0.50\
  --boundary-tolerance-px 32 \
  --save-boundary-debug \
  --images-dir "$IMAGES" \
  --output-dir "$A0_B/evaluation/semantic_validation_debug"
```


### For test datasets evaluate 

``` shell
python gis/evaluate_gis_semantics.py \
  --prediction-dir "$A0_B/test/ours_30000/gis_semantics/probability_u16" \
  --reference-dir "$MANUAL_MASKS" \
  --include-file "$DATA/eval_test.txt" \
  --threshold 0.50\
  --boundary-tolerance-px 32 \
  --save-boundary-debug \
  --images-dir "$IMAGES" \
  --output-dir "$A0_B/evaluation/semantic_test_debug"
```