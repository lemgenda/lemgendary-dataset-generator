# LemGendizedMirNetExposureLarge

> Standardized dataset for image exposure correction and low-light enhancement models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 11,142
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Correct under/over-exposed images and enhance visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Adobe Fivek** | 6,886 | 914 | 7,800 samples |
| **DPED (Smartphone Photography)** | 2,951 | 391 | 3,342 samples |

## Model Training Profiles

### Model: LemGendary MIRNet v2 Exposure Correction

- **Architecture**: MIRNet_v2 (Multi-Scale Residual Network)
- **Optimization**: l1_lpips

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~20.7 dB | > 22.8 dB | **> 24.3 dB** |
| **SSIM** | ~0.7392 | > 0.7980 | **> 0.8400** |
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
| **images** | 9,837 | 1,305 |
| **targets** | 9,837 | 1,305 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedmirnetexposurelarge)
