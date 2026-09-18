# LemGendizedMirNetLowLightLarge

> Standardized dataset for image exposure correction and low-light enhancement models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 15,070
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Correct under/over-exposed images and enhance visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **ExDark (Exclusively Dark)** | 6,480 | 883 | 7,363 samples |
| **LOL-v2 Dataset** | 3,149 | 429 | 3,578 samples |
| **Low Light Image Enhancement Datasets** | 2,269 | 309 | 2,578 samples |
| **LOL (Low-Light) Dataset** | 880 | 120 | 1,000 samples |
| **SIDD (Smartphone Image Denoising)** | 282 | 38 | 320 samples |
| **Learning to See in the Dark (SID)** | 204 | 27 | 231 samples |

## Model Training Profiles

### Model: LemGendary MIRNet v2 Low-Light Enhancement

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
| **images** | 13,264 | 1,806 |
| **targets** | 13,264 | 1,806 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedmirnetlowlightlarge)
