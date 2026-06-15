# YOLO11-FR

This repository provides the author-generated code for the manuscript:

**YOLO11-FR: A bridge crack detection method based on frequency-domain fusion and an edge enhancement mechanism**

YOLO11-FR is an improved YOLO11-based object detection method for bridge crack detection. It integrates a **Fused Fourier Conv Mixer (FFCM)** and a **Residual Edge Enhancement Module (REEM)** to enhance crack-oriented feature representation under complex concrete surface backgrounds.

## Overview

Bridge cracks are usually thin, irregular, and easily confused with concrete texture, stains, shadows, and other surface defects. YOLO11-FR was developed to improve crack detection performance by introducing two task-oriented modules:

* **FFCM**: a frequency-domain feature fusion module for refining high-level crack features.
* **REEM**: an edge enhancement module for strengthening crack boundary and structural representations at multiple detection scales.

This repository includes the modified YOLO11 source code, model configuration files, and dataset conversion scripts used in the manuscript.

## Repository structure

```text
YOLO11-FR/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── requirements.txt
├── docs/
│   └── dataset_preparation.md
├── tools/
│   ├── make_gyu_crack_dataset.py
│   └── convert_crack500_to_yolo.py
└── ultralytics/
    ├── cfg/
    │   └── models/
    │       └── 11/
    │           ├── yolo11_FFCM.yaml
    │           ├── yolo11_REEM.yaml
    │           ├── yolo11_REEM_noStripe.yaml
    │           └── yolo11_FFCM_REEM.yaml
    └── nn/
        ├── tasks.py
        └── modules/
            ├── FFCM.py
            ├── REEM.py
            ├── REEM_noStripe.py
            ├── __init__.py
            ├── block.py
            ├── conv.py
            └── head.py
```

## Environment

The experiments in the manuscript were conducted using the following environment:

```text
Python 3.9.25
PyTorch 2.5.1
CUDA 12.1
Ultralytics 8.3.243
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

To ensure that the modified local `ultralytics/` source code is used, run commands from the root directory of this repository.

## Model configurations

The YOLO11-FR model configuration files are located in:

```text
ultralytics/cfg/models/11/
```

The main configuration files include:

```text
yolo11_FFCM.yaml
yolo11_REEM.yaml
yolo11_REEM_noStripe.yaml
yolo11_FFCM_REEM.yaml
```

The full YOLO11-FR model corresponds to:

```text
ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml
```

## Custom modules

The main custom modules are implemented in:

```text
ultralytics/nn/modules/
```

Key files include:

```text
FFCM.py
REEM.py
REEM_noStripe.py
```

The model parsing logic is integrated through:

```text
ultralytics/nn/tasks.py
```

## Dataset preparation

This repository does not redistribute the original datasets.

The experiments use:

1. **GYU-DET-Crack**, extracted from the public GYU-DET bridge surface defect dataset.
2. **Crack500**, converted from pixel-level crack masks to YOLO-format bounding-box annotations.

Please download the original datasets from their public sources cited in the manuscript.

More details are provided in:

```text
docs/dataset_preparation.md
```

### GYU-DET-Crack construction

The script for constructing the GYU-DET-Crack subset is:

```text
tools/make_gyu_crack_dataset.py
```

Before running the script, modify the dataset paths in the file according to your local directory structure.

### Crack500 conversion

The Crack500 conversion script is:

```text
tools/convert_crack500_to_yolo.py
```

This script converts Crack500 binary masks into YOLO-format bounding-box labels by extracting connected crack regions and generating bounding boxes.

Before running the script, modify the following paths in the file:

```python
ROOT = Path(r"path/to/CRACK500")
OUT = Path(r"path/to/CRACK500_YOLO")
```

Then run:

```bash
python tools/convert_crack500_to_yolo.py
```

The converted dataset will contain:

```text
CRACK500_YOLO/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── crack500.yaml
```

## Training

Example command for training YOLO11-FR:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml data=path/to/data.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

For GYU-DET-Crack:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml data=path/to/GYU_DET_Crack.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

For Crack500:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml data=path/to/CRACK500_YOLO/crack500.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

Automatic mixed precision is disabled during training by setting `amp=False`, which is recommended for stable training of the REEM-related models in this repository.

## Evaluation

After training, the model can be evaluated with:

```bash
yolo detect val model=path/to/best.pt data=path/to/data.yaml imgsz=640
```

Example:

```bash
yolo detect val model=path/to/best.pt data=path/to/GYU_DET_Crack.yaml imgsz=640
```

The main evaluation metrics used in the manuscript include:

```text
Precision
Recall
F1-score
mAP50
mAP50-95
Parameters
GFLOPs
FPS
```

## Ablation experiments

The ablation experiments evaluate the individual and combined effects of FFCM and REEM.

Related configuration files include:

```text
ultralytics/cfg/models/11/yolo11_FFCM.yaml
ultralytics/cfg/models/11/yolo11_REEM.yaml
ultralytics/cfg/models/11/yolo11_REEM_noStripe.yaml
ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml
```

Example command for the REEM variant:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_REEM.yaml data=path/to/GYU_DET_Crack.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

Example command for the FFCM variant:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM.yaml data=path/to/GYU_DET_Crack.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

Example command for the complete YOLO11-FR model:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml data=path/to/GYU_DET_Crack.yaml epochs=100 imgsz=640 batch=16 seed=0 amp=False
```

## Repeated-seed experiments

Repeated-seed experiments were conducted to evaluate the stability of YOLO11-FR.

Example seeds used in the manuscript:

```text
seed 0
seed 1
seed 2
```

Example command:

```bash
yolo detect train model=ultralytics/cfg/models/11/yolo11_FFCM_REEM.yaml data=path/to/GYU_DET_Crack.yaml epochs=100 imgsz=640 batch=16 seed=1 amp=False
```

For another random seed, modify the `seed` value accordingly.

## Notes on pretrained weights

This repository does not include pretrained model weights or trained checkpoint files. Official YOLO11 pretrained weights can be obtained through the Ultralytics framework when required. Trained weights generated during experiments should be saved separately and are not redistributed in this repository.

## Code availability

The author-generated code used for implementing YOLO11-FR, dataset construction, Crack500 conversion, training, evaluation, ablation experiments, and repeated-seed experiments is provided in this repository.

The public datasets used in the study are not redistributed in this repository and should be downloaded from their original public sources cited in the manuscript.

## Citation

If you use this code, please cite the corresponding manuscript:

```text
Zhang Y, Tian B, Guo H. YOLO11-FR: A bridge crack detection method based on frequency-domain fusion and an edge enhancement mechanism.
```

## License

This project is released under the license provided in the `LICENSE` file.

This repository includes modified code based on the Ultralytics YOLO framework. Please also follow the license terms of the original Ultralytics project when using or redistributing this code.
