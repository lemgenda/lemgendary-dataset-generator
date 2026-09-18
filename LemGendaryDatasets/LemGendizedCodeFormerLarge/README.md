# LemGendizedCodeFormerLarge

> Standardized dataset for image restoration models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 22,000
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Restore degraded images and enhance visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Vintage & Degraded Film Archive** | 19,790 | 2,210 | 22,000 samples |

## Model Training Profiles

### Model: LemGendary CodeFormer Face Restoration

- **Architecture**: CodeFormer (Transformer-Based Face Restoration)
- **Optimization**: mse

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **FID** | < 7.80 | < 6.24 | **< 5.2** |
| **PSNR** | ~25.9 dB | > 28.7 dB | **> 30.5 dB** |
| **SSIM** | ~0.8184 | > 0.8835 | **> 0.9300** |
| **LPIPS** | < 0.12 | < 0.10 | **< 0.08** |

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
| **images** | 19,790 | 2,210 |
| **targets** | 19,790 | 2,210 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedcodeformerlarge)
