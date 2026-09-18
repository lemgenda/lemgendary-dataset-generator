# LemGendizedFfaNetIndoorLarge

> Standardized dataset for image dehazing models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 196,304
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Remove haze from images and restore visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Dehazing and Desmoking** | 134,574 | 18,351 | 152,925 samples |
| **RESIDE Standard Indoor** | 12,320 | 1,680 | 14,000 samples |

## Model Training Profiles

### Model: LemGendary FFANet Dehazing (Indoor)

- **Architecture**: BranchedFFANet (Feature Fusion Attention)
- **Optimization**: l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~31.0 dB | > 34.3 dB | **> 36.5 dB** |
| **SSIM** | ~0.8712 | > 0.9405 | **> 0.9900** |
| **LPIPS** | < 0.12 | < 0.10 | **< 0.08** |
| **FID** | < 18.00 | < 14.40 | **< 12.0** |

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
| **images** | 146,894 | 20,031 |
| **targets** | 146,894 | 20,031 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedffanetindoorlarge)
