# LemGendizedNafNetDenoisingLarge

> Standardized dataset for image denoising models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 1,127
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Remove noise from images and restore visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **SIDD (Smartphone Image Denoising)** | 320 | 49 | 369 samples |
| **DND & NAM Noise Data** | 250 | 38 | 288 samples |
| **Multiple Iso Denoising Dataset** | 180 | 28 | 208 samples |
| **9-Classes Noisy Image Dataset** | 100 | 15 | 115 samples |
| **Multi Noise Synthetic Dataset** | 80 | 12 | 92 samples |
| **Salt-and-Pepper Noise** | 47 | 8 | 55 samples |

## Model Training Profiles

### Model: LemGendary NAFNet Denoising

- **Architecture**: NAFNet (Nonlinear Activation-Free Network)
- **Optimization**: l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~34.2 dB | > 37.8 dB | **> 40.2 dB** |
| **SSIM** | ~0.8492 | > 0.9167 | **> 0.9650** |
| **LPIPS** | < 0.03 | < 0.02 | **< 0.02** |
| **FID** | < 6.00 | < 4.80 | **< 4.0** |

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
| **images** | 977 | 150 |
| **targets** | 977 | 150 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizednafnetdenoisinglarge)
