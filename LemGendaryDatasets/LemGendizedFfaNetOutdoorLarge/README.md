# LemGendizedFfaNetOutdoorLarge

> Standardized dataset for image dehazing models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 217,113
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Remove haze from images and restore visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Outdoor Dehazing Dataset** | 190,708 | 26,005 | 216,713 samples |
| **O-HAZE / NTIRE Dehazing** | 159 | 21 | 180 samples |
| **NH-HAZE Dataset** | 97 | 13 | 110 samples |
| **Hazing Images Dataset (CVPR)** | 97 | 13 | 110 samples |

## Model Training Profiles

### Model: LemGendary FFANet Dehazing (Outdoor)

- **Architecture**: BranchedFFANet (Feature Fusion Attention)
- **Optimization**: l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~28.6 dB | > 31.7 dB | **> 33.7 dB** |
| **SSIM** | ~0.8677 | > 0.9367 | **> 0.9860** |
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
| **images** | 191,061 | 26,052 |
| **targets** | 191,061 | 26,052 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedffanetoutdoorlarge)
