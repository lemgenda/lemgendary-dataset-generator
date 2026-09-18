# LemGendizedMprNetDerainingLarge

> Standardized dataset for image deraining models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 34,407
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Remove rain streaks from images and restore visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Balraj Rain Dataset** | 16,304 | 2,276 | 18,580 samples |
| **Rain100L** | 4,800 | 670 | 5,470 samples |
| **Rain100H** | 4,800 | 670 | 5,470 samples |
| **High-Resolution Rainy Images** | 4,289 | 598 | 4,887 samples |

## Model Training Profiles

### Model: LemGendary MPRNet Deraining

- **Architecture**: MPRNet (Multi-Stage Progressive Network)
- **Optimization**: l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~26.0 dB | > 28.8 dB | **> 30.6 dB** |
| **SSIM** | ~0.7920 | > 0.8550 | **> 0.9000** |
| **LPIPS** | < 0.11 | < 0.08 | **< 0.07** |
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
| **images** | 30,193 | 4,214 |
| **targets** | 30,193 | 4,214 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedmprnetderaininglarge)
