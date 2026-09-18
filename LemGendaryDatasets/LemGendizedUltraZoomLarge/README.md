# LemGendizedUltraZoomLarge

> High-fidelity dataset for image super-resolution and ultra-zoom models.

## Dataset Overview

- **Category:** Super-Resolution
- **Total Samples:** 17,724
- **Architecture Base:** Transformer-based or Deep Residual networks
- **Primary Task:** Scale low-resolution images to high-resolution while preserving details.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Df2K Ost** | 12,172 | 1,602 | 13,774 samples |
| **Flickr2K** | 2,349 | 301 | 2,650 samples |
| **Div2K Dataset** | 792 | 108 | 900 samples |
| **Urban100** | 349 | 51 | 400 samples |

## Model Training Profiles

### Model: LemGendary UltraZoom Master Model

- **Architecture**: UltraZoomMaster (Sub-Pixel ESPCN Super-Resolution)
- **Optimization**: mse

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~28.9 dB | > 32.0 dB | **> 34.0 dB** |
| **SSIM** | ~0.8360 | > 0.9025 | **> 0.9500** |
| **LPIPS** | < 0.06 | < 0.05 | **< 0.04** |
| **FID** | < 15.00 | < 12.00 | **< 10.0** |

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
| **images** | 15,662 | 2,062 |
| **targets** | 15,662 | 2,062 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedultrazoomlarge)
