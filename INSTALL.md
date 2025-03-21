## Installation (from Mask2Former)

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## Install Cluster2Former with Apptainer
* The apptainer image SIF file is available [here](https://drive.google.com/uc?id=1F1Si2_Iv-sS-F4uwX_3eplvZSXN_aegg). It has built from `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04`.
* You can find [definition file](build_container.def) of the appatiner image in the repository, if you want to use it.
* You can find the [container_launcher.sh](container_launcher.sh) file, before you run this file in a terminal, you should modify it to start the apptainer instance and possibly make the links to your dataset (see more in the file).
* It is recommended to run the [container_launcher.sh](container_launcher.sh) in a screen terminal.
* After the running, in the apptainer you should activate the virtual environment, what is given in the file.  
* Work with Cluster2Former and Mask2Former
