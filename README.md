# LegRadNet: Multi-Head Attention Ensemble with Optuna Tuning

LegRadNet is a deep learning pipeline for classifying knee radiographs (such as Kellgren-Lawrence grades 0–4). It merges **five powerful backbones**—**DenseNet201, Swin Transformer, ConvNeXt, Vision Transformer (ViT), and Xception**—using **multi-head attention** gating. This approach captures both **local CNN features** and **global Transformer context** for robust classification. We also include **Optuna** for hyperparameter tuning.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Model Descriptions](#model-descriptions)  
3. [Multi-Head Attention Gating](#multi-head-attention-gating)  
4. [Repository Structure](#repository-structure)  
5. [Installation](#installation)  
6. [Configuration](#configuration)  
7. [Data Preparation](#data-preparation)  
8. [Training](#training)  
9. [Inference](#inference)  
10. [Grad-CAM Visualization](#grad-cam-visualization)  
11. [Optuna Hyperparameter Tuning](#optuna-hyperparameter-tuning)  
12. [Future Work](#future-work)  
13. [License](#license)
---

## Introduction

LegRadNet aims to **classify knee radiographs** into five categories. Typical usage scenarios include:

- **Kellgren-Lawrence grading** (0–4) for osteoarthritis severity.  
- Other 5-class orthopedic or musculoskeletal classification tasks.

By leveraging **multiple backbones**—a mix of CNNs and Transformers—LegRadNet can capture both **fine local details** and **global structure**. A **multi-head attention gating** network fuses these backbone embeddings into a single representation.

### Example: Kellgren-Lawrence Grading of Knee Osteoarthritis
![image](https://github.com/user-attachments/assets/24101b31-d6fe-41a5-ab67-d41f0a557aff)

*(Image from [SAGE Journals](https://journals.sagepub.com/doi/full/10.1177/2325967120927481#body-ref-bibr2-2325967120927481), illustrating KL grades 0–4.)*

---

## Model Descriptions

LegRadNet relies on **five** distinct backbone models. Each produces a `feature_dim`-sized vector, which is later fused in the ensemble.

### 1. DenseNet201
- **Core Idea**: Dense connectivity, where each layer receives inputs from all preceding layers.  
- **Why It Helps**: Improves gradient flow and reuses features, crucial for subtle medical image cues.  
- **Implementation**: Uses `torchvision.models.densenet201`, removing the default classifier and projecting to `feature_dim`.

### 2. Swin Transformer
- **Core Idea**: A hierarchical Transformer using **shifted window** self-attention.  
- **Why It Helps**: Captures global relationships efficiently by sliding local windows over the image.  
- **Implementation**: Via `timm`, e.g. `"swin_base_patch4_window7_224"`, discarding the classification head and replacing it with a projection layer.

### 3. ConvNeXt
- **Core Idea**: A modern CNN incorporating design elements from Transformers (large kernel sizes, simplified blocks).  
- **Why It Helps**: Maintains CNN inductive biases while streamlining architecture for improved performance.  
- **Implementation**: Uses `create_model("convnext_base")`, removing the classifier, final embedding mapped to `feature_dim`.

### 4. Vision Transformer (ViT)
- **Core Idea**: Splits the image into patches, processes them as tokens in a standard Transformer pipeline.  
- **Why It Helps**: Captures **global dependencies** across patches, beneficial for large-scale structural analysis.  
- **Implementation**: Takes the [CLS] token from a `vit_base_patch16_224` model, projecting it to `feature_dim`.

### 5. Xception
- **Core Idea**: Extensively uses **depthwise separable convolutions**, an “extreme” Inception variant.  
- **Why It Helps**: Reduces parameter count and can be more efficient for large images.  
- **Implementation**: Uses `timm`'s Xception model, dropping the default head, and adding a linear layer for `feature_dim`.

---

## Multi-Head Attention Gating

To fuse the feature vectors from all backbones, LegRadNet applies **multi-head attention**:

1. **Stack Features**: Suppose `N` backbones, each output `[B, feature_dim]`. We stack to `[B, N, feature_dim]`.  
2. **Learnable Query**: A parameter `[1, 1, feature_dim]` is expanded to `[B, 1, feature_dim]`.  
3. **Attention**: The query attends over the `N` tokens (backbone features) via multi-head attention.  
4. **Fused Output**: Produces a single `[B, feature_dim]` embedding that a final classifier maps to 5 classes.

### Flow Diagram Representation
<img width="1037" alt="image" src="https://github.com/user-attachments/assets/ff5a9ddf-65f7-47e7-a9d1-77758f73f080" />
*(This diagram shows how different backbones contribute embeddings that are fused using multi-head attention.)*

---

## Repository Structure

```
LegRadNet/
├── requirements.txt
├── configs/
│   └── ensemble_config.yaml
├── models/
│   ├── densenet.py
│   ├── swin_transformer.py
│   ├── convnext.py
│   ├── vit.py
│   ├── xception.py
│   ├── gating_network.py
│   └── ensemble_model.py
├── scripts/
    ├── data_loader.py
    ├── preprocessing.py
    ├── utils.py
    ├── grad_cam.py
    ├── train.py
    ├── inference.py
    └── hparam_tuning.py

```

---

## Installation

1. **Clone and Enter**:
   ```bash
   git clone https://github.com/PersonX132/LegRadNet.git
   cd LegRadNet
   ```
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---
## Configuration

Edit `configs/ensemble_config.yaml` to set paths, model options, hyperparams. For example:

```yaml
model:
  backbones:
    - densenet
    - swin_transformer
    - convnext
    - vit
    - xception
  gating_network: true
  gating_mode: "multi_head_attention"  # "multi_head_attention" or "simple_mlp"
  feature_dim: 1024
  num_classes: 5
  pretrained: true

hyperparameter_tuning:
  enable_optuna: true
  n_trials: 10
  param_spaces:
    learning_rate:
      low: 1e-5
      high: 1e-3
    weight_decay:
      low: 1e-5
      high: 1e-2
```

---

## Data Preparation

1. Place your knee radiograph images in a directory (e.g., `data/images/`).  
2. Create a CSV file (e.g., `my_data.csv`) with:
   ```
   filename,label
   knee1.jpg,0
   knee2.jpg,4
   ...
   ```
   The `filename` should be relative to `data.dataset_path`.

3. Update `dataset_path` and `csv_path` in `configs/ensemble_config.yaml`.  


---

## Training

Run:
```bash
python scripts/train.py --config ./configs/ensemble_config.yaml
```

---

## Inference

Use a trained checkpoint to classify a single image:
```bash
python scripts/inference.py \
  --config ./configs/ensemble_config.yaml \
  --checkpoint ./checkpoints/checkpoint_epochX.pth \
  --image_path ./data/images/test_knee.jpg
```

---

## Grad-CAM Visualization

To inspect how CNN backbones focus on specific regions:
```bash
python scripts/grad_cam.py \
  --config ./configs/ensemble_config.yaml \
  --checkpoint ./checkpoints/checkpoint_epochX.pth \
  --image_path ./data/images/test_knee.jpg \
  --output_path gradcam_output.jpg
```
### Example of Grad-CAM Output
![image](https://github.com/user-attachments/assets/ee25d0b5-6e26-46b7-b9c4-05d6e3ce1a0e)


*(Based on methods discussed in [Radiology Journal](https://pubs.rsna.org/doi/abs/10.1148/radiol.2020192091?journalCode=radiology), this is an example of what a Grad-CAM visualization might look like.)*

---

## Optuna Hyperparameter Tuning

We provide a script for **Optuna**-based hyperparameter search:
```bash
python scripts/hparam_tuning.py --config ./configs/ensemble_config.yaml
```

---

## Future Work

1. **Advanced Explainability**  
   - Extend Grad-CAM or attention-based visualization to the Swin Transformer and ViT backbones.
2. **New Backbones**  
   - Integrate other CNN/Transformer architectures (e.g., EfficientNet, DeiT, BEiT) to see if they complement the existing set.
3. **Enhanced Data Augmentation**  
   - Explore domain-specific transformations for knee radiographs, or apply self-supervised methods on unlabeled medical images.
4. **Attention Visualization**  
   - For multi-head attention gating, a script could map how each backbone is weighted across multiple images.

## License

This project is licensed under the **MIT License**.

This means **anyone** can:
- **Use** this software for any purpose (personal, academic, commercial, etc.).
- **Modify** and adapt it freely.
- **Distribute** it, including selling or incorporating it into other projects.
- **Sublicense** it under a different license if desired.

The **only requirement** is that the original copyright notice is included in all copies or substantial portions of the software.

This software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the author be liable for any claims, damages, or other liabilities.

For full details, see the **[LICENSE](LICENSE)** file.
