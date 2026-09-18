# LemGendizedUpnV2Large

> High-precision dataset for camera parameter prediction, tone mapping, and photographic exposure estimation.

## Dataset Overview

- **Category:** Photographic Parameter Prediction
- **Total Samples:** 1,378,070
- **Architecture Base:** Deep Multi-Layer Perceptron / Convolutional Regressor
- **Primary Task:** Predict global and local tone adjustment parameters from input images.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **DPED (Smartphone Photography)** | 1,277,236 | 67,284 | 1,344,520 samples |
| **Adobe FiveK** | 28,447 | 1,553 | 30,000 samples |
| **Flickr** | 2,537 | 113 | 2,650 samples |
| **DIV2K Dataset** | 853 | 47 | 900 samples |

## Model Training Profiles

### Model: LemGendary UPN v2 Parameter Predictor

- **Architecture**: UPN_v2 (MobileNet-Lite Parameter Regressor)
- **Optimization**: smooth_l1

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Mae** | < 0.08 | < 0.06 | **< 0.05** |

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
| **images** | 1,309,073 | 68,997 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedupnv2large)
