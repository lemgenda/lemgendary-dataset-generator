# LemGendizedNafNetDebluringLarge

> Standardized dataset for image deblurring models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 6,679
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Remove blur from images and restore visual sharpness.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **GoPro Deblurring Dataset** | 2,800 | 388 | 3,188 samples |
| **RealBlur Dataset** | 1,750 | 242 | 1,992 samples |
| **HiDeBlur Dataset** | 880 | 122 | 1,002 samples |
| **Image Deblurring Performance** | 437 | 60 | 497 samples |

## Model Training Profiles

### Model: LemGendary NAFNet Debluring

- **Architecture**: NAFNet (Nonlinear Activation-Free Network)
- **Optimization**: l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~28.8 dB | > 31.9 dB | **> 33.9 dB** |
| **SSIM** | ~0.8536 | > 0.9215 | **> 0.9700** |
| **LPIPS** | < 0.06 | < 0.05 | **< 0.04** |
| **FID** | < 9.00 | < 7.20 | **< 6.0** |

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
| **images** | 5,867 | 812 |
| **targets** | 5,867 | 812 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizednafnetdebluringlarge)
