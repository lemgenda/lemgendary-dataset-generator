# LemGendizedProfessionalMultitaskRestorationLarge

> Standardized dataset for image restoration models.

## Dataset Overview

- **Category:** Image Restoration
- **Total Samples:** 343,911
- **Architecture Base:** Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network
- **Primary Task:** Restore degraded images and enhance visual quality.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **ffanetoutdoor Multi-Task Sub-Manifold** | 140,791 | 19,229 | 160,020 samples |
| **parsenet Multi-Task Sub-Manifold** | 58,368 | 7,930 | 66,298 samples |
| **ffanetindoor Multi-Task Sub-Manifold** | 41,258 | 5,530 | 46,788 samples |
| **mprnetderaining Multi-Task Sub-Manifold** | 30,193 | 4,214 | 34,407 samples |
| **codeformer Multi-Task Sub-Manifold** | 11,832 | 1,575 | 13,407 samples |
| **mirnetexposure Multi-Task Sub-Manifold** | 9,837 | 1,305 | 11,142 samples |
| **nafnetdebluring Multi-Task Sub-Manifold** | 5,867 | 812 | 6,679 samples |
| **ultrazoom Multi-Task Sub-Manifold** | 1,666 | 239 | 1,905 samples |
| **mirnetlowlight Multi-Task Sub-Manifold** | 1,047 | 163 | 1,210 samples |
| **nafnetdenoising Multi-Task Sub-Manifold** | 977 | 150 | 1,127 samples |
| **filmrestorer Multi-Task Sub-Manifold** | 828 | 100 | 928 samples |

## Model Training Profiles

### Model: LemGendary Professional Multi-Task Restoration Model

- **Architecture**: MultiTaskRestorer (Shared Encoder Multi-Task MoE)
- **Optimization**: hybrid_restoration

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **PSNR** | ~27.2 dB | > 30.1 dB | **> 32.0 dB** |
| **SSIM** | ~0.8184 | > 0.8835 | **> 0.9300** |
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
| **images** | 302,664 | 41,247 |
| **targets** | 302,664 | 41,247 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedprofessionalmultitaskrestorationlarge)
