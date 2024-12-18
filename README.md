# ResNet-18_CBAM for Digital Cytopathology Image Classification

This repository provides the implementation of ResNet-18_CBAM network for digital cytopathology image classification. 


## Overview

### Model Architecture

![Network Architecture](https://raw.githubusercontent.com/cir9/resnet-cbam-py/refs/heads/main/images/arch.png)

---

### Experimental Dataset
The dataset used in this study originates from the Department of Pathology at a domestic hospital, comprising cytopathology images collected through Rapid On-Site Evaluation (ROSE) during fine needle aspiration of pulmonary masses. All images are stained with Diff-Quik, captured at 400x magnification with a resolution of 50 µm/pixel, and stored as JPEG files with dimensions of 1916x1010 at 96 dpi.

#### Cell Types:
The dataset contains six types of lung cells:
1. **Adenocarcinoma:** Irregular gland structures, increased nuclear-to-cytoplasmic ratio, and disordered arrangements.
2. **Squamous Cell Carcinoma:** Polygonal or irregular shapes with distinct cell boundaries and keratinization.
3. **Carcinoid:** Uniform medium-sized cells with regular nuclear shapes and slower growth.
4. **Non-Small Cell Lung Cancer (NSCLC):** Morphological diversity excluding the characteristics of small cell lung cancer.
5. **Small Cell Lung Cancer (SCLC):** Small cell size, dense nuclei, high nuclear-to-cytoplasmic ratio, and frequent mitoses.
6. **Normal Lung Cells:** Regular cell size and shape, low nuclear-to-cytoplasmic ratio, and absence of atypia.


#### Dataset Composition:
- **Total Cases:** 306
  - Adenocarcinoma: 80
  - Squamous Cell Carcinoma: 27
  - Carcinoid: 38
  - NSCLC: 27
  - SCLC: 55
  - Normal: 79
- **Split:**
  - Training Set: 259 cases
  - Test Set: 47 cases


## Environment and Dependencies
- **Hardware:** 
  - GPU: NVIDIA 2070 super (8GB VRAM)
  - CPU: AMD R7-3700X
  - Operating System: Windows 10
- **Software:**
  - Python 3.9.7
  - PyTorch 1.10.2
  - SimpleITK 2.1.1
  - NumPy 1.21.5

## Model Training
-  **Cross-validation:** 5-fold
-  **Batch Size:** 2
-  **Optimizer:** Adam
-  **Initial LR:** 0.0001
-  **LR Schedule:** Reduced by 1/10 every 15 epochs
-  **Epochs:** 150
