<img src="imgs/nnInteractive_header_white.png">

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
import torch
import SimpleITK as sitk
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed

# --- Download Trained Model Weights (~400MB) ---
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = "/home/isensee/temp"  # Specify the download directory

download_path = snapshot_download(
    repo_id=REPO_ID, 
    allow_patterns=[f"{MODEL_NAME}/*"],
    local_dir=DOWNLOAD_DIR
)

# The model is now stored in DOWNLOAD_DIR/MODEL_NAME.

# --- Initialize Inference Session ---
from nnInteractive.inference.nnInteractiveInferenceSessionV3 import nnInteractiveInferenceSessionV3

session = nnInteractiveInferenceSessionV3(
    device=torch.device("cuda:0"),  # Set inference device
    use_torch_compile=False,  # Experimental: Not tested yet
    verbose=False,
    torch_n_threads=os.cpu_count(),  # Use available CPU cores
    interaction_decay=0.9,  # Must match the training setting
    use_background_preprocessing=True,  # Enables threading for preprocessing and memory allocation (useful in GUI mode)
    do_prediction_propagation=True,  # Enables AutoZoom for better patching
    use_pinned_memory=True,  # Optimizes GPU memory transfers
    verbose_run_times=False,
)

# Load the trained model
model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
session.initialize_from_trained_model_folder(model_path)

# --- Load Input Image (Example with SimpleITK) ---
input_image = sitk.ReadImage("FILENAME")
img = sitk.GetArrayFromImage(input_image)[None]  # Ensure shape (1, x, y, z)

# Validate input dimensions
if img.ndim != 4:
    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

session.set_image(img)

# --- Define Output Buffer ---
target_tensor = torch.zeros(img.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
session.set_target_buffer(target_tensor)

# --- Interacting with the Model ---
# Interactions can be freely chained and mixed in any order. Each interaction refines the segmentation.
# The model updates the segmentation mask in the target buffer after every interaction.

# Example: Add a point interaction
# POINT_COORDINATES should be a tuple (x, y, z) specifying the point location.
session.add_point_interaction(POINT_COORDINATES, include_interaction=True)

# Example: Add a bounding box interaction
# BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
# Note: nnInteractive pre-trained models currently only support **2D bounding boxes**.
# This means that **one dimension must be [d, d+1]** to indicate a single slice.

# Example of a 2D bounding box in the axial plane (XY slice at depth Z)
# BBOX_COORDINATES = [[30, 80], [40, 100], [10, 11]]  # X: 30-80, Y: 40-100, Z: slice 10

session.add_bbox_interaction(BBOX_COORDINATES, include_interaction=True)

# Example: Add a scribble interaction
# - A 3D image of the same shape as img where one slice (any axis-aligned orientation) contains a hand-drawn scribble.
# - Background must be 0, and scribble must be 1.
# - Use session.preferred_scribble_thickness for optimal results.
session.add_scribble_interaction(SCRIBBLE_IMAGE, include_interaction=True)

# Example: Add a lasso interaction
# - Similarly to scribble a 3D image with a single slice containing a **closed contour** representing the selection.
session.add_lasso_interaction(LASSO_IMAGE, include_interaction=True)

# You can combine any number of interactions as needed. 
# The model refines the segmentation result incrementally with each new interaction.

# --- Retrieve Results ---
# The target buffer holds the segmentation result.
results = session.target_buffer.clone()
# OR (equivalent)
results = target_tensor.clone()

# Cloning is required because the buffer will be **reused** for the next object.
# Alternatively, set a new target buffer for each object:
session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))

# --- Start a New Object Segmentation ---
session.reset_interactions()  # Clears the target buffer and resets interactions

# Now you can start segmenting the next object in the image.

# --- Set a New Image ---
# Setting a new image also requires setting a new matching target buffer
session.set_image(NEW_IMAGE)
session.set_target_buffer(torch.zeros(NEW_IMAGE.shape[1:], dtype=torch.uint8))

# Enjoy!
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