# LemGendizedNimaAestheticLarge

> Unified dataset for training SOTA quality models and evaluation systems.

## Dataset Overview

- **Category:** Image Quality Assessment
- **Total Samples:** 321,369
- **Architecture Base:** Deep Convolutional Network / Vision Transformer with Earth Mover's Distance Optimization
- **Primary Task:** Predict human-perceptual quality score.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **AVA** | 216,675 | 29,614 | 246,289 samples |
| **Tad66K For Image Aesthetics Assessment** | 48,386 | 6,579 | 54,965 samples |
| **SPAQ** | 9,787 | 1,333 | 11,120 samples |
| **KonIQ-10k** | 7,925 | 1,070 | 8,995 samples |

## Model Training Profiles

### Model: LemGendary NIMA Aesthetic Scorer (Mobile)

- **Architecture**: MobileNetV2 (Global Composition)
- **Optimization**: emd

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Plcc** | ~0.52 | > 0.59 | **> 0.65** |
| **Srcc** | ~0.52 | > 0.59 | **> 0.65** |

### Model: LemGendary NIMA Aesthetic Scorer (EfficientNetV2-S)

- **Architecture**: EfficientNetV2-S (Global Composition)
- **Optimization**: emd

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Plcc** | ~0.56 | > 0.63 | **> 0.7** |
| **Srcc** | ~0.56 | > 0.63 | **> 0.7** |

### Model: LemGendary NIMA Aesthetic Scorer (Pro ViT)

- **Architecture**: Swin-v2-T (Global Multi-Scale Attention)
- **Optimization**: emd

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Plcc** | ~0.60 | > 0.68 | **> 0.75** |
| **Srcc** | ~0.60 | > 0.68 | **> 0.75** |

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

- **`category.txt`**: Top-level categorization tag.
- **`classes.txt`**: Class labels mapping.
- **`dataset-metadata.json`**: Kaggle Frictionless metadata manifest, licensing, and schema column definitions for Parquet feature tensors.
- **`dataset_info.yaml`**: Manifest metadata for automated PyTorch loaders.
- **`README.md`**: This documentation file.

## Physical Data Manifest

| Folder | Train | Val |
| :--- | :--- | :--- |
| **images** | 282,773 | 38,596 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaaestheticlarge)
