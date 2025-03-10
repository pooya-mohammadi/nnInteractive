LOGO

# nnInteractive: Redefining 3D Promptable Segmentation

This repository contains the nnInteractive python backend for our 
[napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive) and [MITK integration](Todo). It can be used for 
python-based inference.


## What is nnInteractive?

    paper

##### Abstract:

Accurate and efficient 3D segmentation is essential for both clinical and research applications. While foundation 
models like SAM have revolutionized interactive segmentation, their 2D design and domain shift limitations make them 
ill-suited for 3D medical images. Current adaptations address some of these challenges but remain limited, either 
lacking volumetric awareness, offering restricted interactivity, or supporting only a small set of structures and 
modalities. Usability also remains a challenge, as current tools are rarely integrated into established imaging 
platforms and often rely on cumbersome web-based interfaces with restricted functionality. We introduce nnInteractive, 
the first comprehensive 3D interactive open-set segmentation method. It supports diverse prompts—including points, 
scribbles, boxes, and a novel lasso prompt—while leveraging intuitive 2D interactions to generate full 3D 
segmentations. Trained on 120+ diverse volumetric 3D datasets (CT, MRI, PET, 3D Microscopy, etc.), nnInteractive 
sets a new state-of-the-art in accuracy, adaptability, and usability. Crucially, it is the first method integrated 
into widely used image viewers (e.g., Napari, MITK), ensuring broad accessibility for real-world clinical and research 
applications. Extensive benchmarking demonstrates that nnInteractive far surpasses existing methods, setting a new 
standard for AI-driven interactive 3D segmentation.

<img src="imgs/figure1_method.png" width="1200">


## Installation

### Prerequisites

You need a Linux or Windows computer with a Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with \<6GB.

##### 1. Create a virtual environment:

nnInteractive supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n nnInteractive python=3.12
conda activate nnInteractive
```

##### 2. Install the correct PyTorch for your system

Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the right configuration.
Note that since recently PyTorch needs to be installed via pip. This is fine to do within your conda environment.

For Ubuntu with a Nvidia GPU, pick 'stable', 'Linux', 'Pip', 'Python', 'CUDA12.6' (if all drivers are up to date, otherwise use and older version):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### 3. Install this repository
Either install via pip:
`pip install nninteractive_inference`

Or clone and install this repository:
```bash
git clone https://github.com/MIC-DKFZ/nnInteractive_inference
cd nnInteractive_inference
pip install -e .
```

## Getting Started
Here is a minimalistic script that covers the core functionality of nnInteractive:

```python
import os
from huggingface_hub import snapshot_download  # you might have to install huggingface_hub manually as this is not part of the repository dependencies
import torch

# download trained model weights (~400MB)
repo_id = "nnInteractive/nnInteractive"
model_name = "nnInteractive_v1.0"  # there might be updated models in the future

download_path = snapshot_download(
    repo_id=repo_id, allow_patterns=[f"{model_name}/*"]
)

# the downloaded model is now in XXXXX.

# Initialize an inference session. The downloaded model will tell you what the preferred inference session is as part of its YYYY file. 
from nnInteractive.inference.nnInteractiveInferenceSessionV3 import nnInteractiveInferenceSessionV3

session = nnInteractiveInferenceSessionV3(
    device=torch.device('cuda:0'),
    use_torch_compile=False,  # not tested for now
    verbose=False,
    torch_n_threads=os.cpu_count(),
    interaction_decay=0.9, # same as in training
    use_background_preprocessing=True, # use threading for preprocessing and memory allocation. Primarily useful in the GUI
    do_prediction_propagation=True, # AutoZoom
    use_pinned_memory=True,
    verbose_run_times=False
)

# set an image, here an example for how to load one with SimpleITK
import SimpleITK as sitk
img = sitk.GetArrayFromImage(sitk.ReadImage('FILENAME'))

# Image MUST be 4D with shape (1, x, y, z)
session.set_image(img)

# give the session a target array to write results into. Must be 3D (x, y, z)
target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)
session.set_target_buffer(target_tensor)

# now we are ready to interact! Here are some example prompts
# IMPORTANT: include_interaction indicates whether this is a positive (True) or negative (False) prompt
# The session will run a prediction after each new prompt

# point prompt. Point coordinates are a tuple (a, b, c) of where in the image the point should be placed
session.add_point_interaction(POINT_COORDINATES, include_interaction=True)

# bbox prompt: Bboxes are half-open intervals [a, b) in each dimension: [[x1, x2], [y1, y2], [z1, z2]]. This function is used for 2D and for 3D bounding boxes.
# IMPORTANT: nnInteractive pretrained models only support 2D bounding boxes! So one of the dimensions must be [x1, x1 + 1]
session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)

# scribble prompt: A 3D image of the same shape as img that has a hand drawn scribble in it. Background must be 0, scribble must be 1!
# since scribbles are 2D only one slice in any orientation is populated in the 3D image
# nnInteractive models expect a certain scribble thickness. Use session.preferred_scribble_thickness!
session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# lasso prompt. Just like scribbles we expect a 3D image in here where in one slice (any plane) there is a closed contour depicting the lasso selection.
session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# Results can be retrieved at any time. They are written in the target_tensor:
# Either
results = session.target_buffer 
# OR
results = target_tensor

```


## Citation
When using nnInteractive, please cite the following paper:

    todo

## Acknowledgments

<p align="left">
  <img src="imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).