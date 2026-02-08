# XAI_DogVSCat
Explainable Dog vs Cat Classification via Part-Grounded Activations

---

## Overview
This project studies **Explainable AI (XAI)** for **dog vs. cat image classification** by grounding model explanations to **object parts** (ears, eyes, nose, mouth).

Unlike segmentation-based approaches, this project:
- performs **classification on raw images**,
- uses **part-level bounding boxes only** (no segmentation masks),
- explains predictions via **CNN activation patterns**,
- and models salient regions using a **Variational Autoencoder (VAE)**.

The main question addressed is:
> *Which object parts does the classifier rely on when distinguishing dogs from cats?*

---

## Key Ideas
- **Classification-first**: the classifier is trained purely on raw images.
- **Part grounding**: part bounding boxes are generated automatically and used only for explanation.
- **Activation-space explainability**: explanations operate on CNN feature maps rather than pixels.
- **Generative region discovery**: a VAE is used to discover salient regions in activation space.

---

## Assumption
This project implicitly relies on the **local spatial smoothness of convolutional feature maps**.

In practice, this assumes a **locally Lipschitz image-to-feature mapping**, meaning:
- small changes in the input image lead to smooth, localized changes in activation maps,
- enabling meaningful region discovery and part alignment.

This assumption naturally holds for CNNs (e.g., ResNet) and motivates the architectural choice.

---

## Pipeline

### 1. Dataset
- Input: raw RGB images
- Categories:
  - `dog`
  - `cat`
- Example datasets:
  - Kaggle Dogs vs Cats
  - Oxford-IIIT Pet

No part annotations or segmentation masks are required.

---

### 2. Part Bounding Box Generation (Auxiliary)
Part-level bounding boxes are **automatically generated** using **GroundingDINO**.

For each image:
1. The main category (`dog` or `cat`) is detected.
2. Category-specific part prompts are queried:
   - `left_eye`, `right_eye`
   - `nose`
   - `mouth`
   - `left_ear`, `right_ear`
3. Boxes are filtered, deduplicated, and optionally expanded.

These bounding boxes are **not used to train the classifier**.  
They serve only as **semantic grounding for explanations**.

---

### 3. Classification (CNN)
- Backbone: **ResNet (CNN)**
- Input: raw image
- Output: class probability (`dog` / `cat`)

CNNs are chosen because they:
- preserve spatial structure,
- produce stable activation maps,
- support reliable region-based explainability (e.g., Grad-CAM).

---

### 4. Activation Extraction
Spatial feature maps are extracted from the CNN:
- typically from the final convolutional layer (before global pooling),
- producing activation tensors of shape `C × H × W`.

These activations form the basis for all explainability analyses.

---

### 5. Explainability Baseline
As a sanity check, standard explainability methods are applied:
- Grad-CAM / CAM
- Heatmaps are compared against detected part bounding boxes

This provides a strong baseline before introducing generative modeling.

---

### 6. VAE-Based Region Discovery
A **Variational Autoencoder (VAE)** is trained on activation maps to model their latent structure.

Two usage modes are supported:
- **Reconstruction-based saliency**: reconstruction error highlights important regions.
- **Mask / heatmap generation**: the decoder outputs a sparse, smooth saliency map.

The VAE identifies **salient regions in activation space**, which are:
- projected back to image space,
- converted to bounding boxes,
- aligned with detected object parts.

---

### 7. Part-Grounded Explanation
Final explanations are obtained by:
- measuring overlap between VAE-derived regions and part bounding boxes,
- reporting which parts contribute most to the classification decision.

This yields **semantically grounded explanations**, such as:
> “The model relies primarily on ear and face regions when classifying dogs.”

---

## What This Project Does NOT Do
- ❌ No segmentation masks
- ❌ No pixel-level supervision
- ❌ No part annotations during classifier training
- ❌ No scene-level reasoning

---

## Installation (Example)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install -r requirements.txt
# XAI_DogVSCat
