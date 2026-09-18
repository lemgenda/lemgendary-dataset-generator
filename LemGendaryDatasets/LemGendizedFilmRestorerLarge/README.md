# LemGendizedFilmRestorerLarge

> Standardized dataset for image old film restoration models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 67,542
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Restore degraded vintage film frames (scratches, noise, color fade).

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Vintage & Degraded Film Archive** | 56,210 | 7,690 | 63,900 samples |
| **Vintage Degraded Photos** | 1,235 | 333 | 1,568 samples |
| **DIV2K Dataset** | 1,036 | 242 | 1,278 samples |
| **SIDD (Smartphone Image Denoising)** | 508 | 132 | 640 samples |
| **DND & NAM Noise Data** | 93 | 21 | 114 samples |
| **Photo Restoration Dataset** | 33 | 9 | 42 samples |

## Model Training Profiles

### Model: LemGendary Universal Film Restorer

- **Architecture**: UniversalFilmRestorer (Residual Dense Autoencoder)
- **Optimization**: l1_lpips

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~20.4 dB | > 22.6 dB | **> 24.0 dB** |
| **SSIM** | ~0.7040 | > 0.7600 | **> 0.8000** |
| **LPIPS** | < 0.38 | < 0.30 | **< 0.25** |
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
| **images** | 59,115 | 8,427 |
| **targets** | 59,115 | 8,427 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedfilmrestorerlarge)
