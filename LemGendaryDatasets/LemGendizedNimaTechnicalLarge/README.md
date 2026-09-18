# LemGendizedNimaTechnicalLarge

> Unified dataset for training SOTA quality models and evaluation systems.

## Dataset Overview

- **Category:** Image Quality Assessment
- **Total Samples:** 26,093
- **Architecture Base:** Deep Convolutional Network / Vision Transformer with Earth Mover's Distance Optimization
- **Primary Task:** Predict human-perceptual quality score.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **SPAQ** | 8,862 | 2,258 | 11,120 samples |
| **KonIQ-10k** | 8,289 | 2,076 | 10,365 samples |
| **TID2013** | 2,138 | 506 | 2,644 samples |
| **LIVE** | 790 | 221 | 1,011 samples |
| **CSIQ** | 721 | 175 | 896 samples |
| **DND & NAM Noise Data** | 46 | 11 | 57 samples |

## Model Training Profiles

### Model: LemGendary NIMA Technical Scorer

- **Architecture**: EfficientNetV2-S (Spatial Integrity)
- **Optimization**: emd

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Plcc** | ~0.73 | > 0.82 | **> 0.91** |
| **Srcc** | ~0.73 | > 0.82 | **> 0.91** |
| **Rank Margin** | < 0.08 | < 0.06 | **< 0.05** |

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
| **images** | 20,846 | 5,247 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimatechnicallarge)
