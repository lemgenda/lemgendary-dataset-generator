# LemGendizedNimaAuthenticityLarge

> Unified dataset for training SOTA models to distinguish between AI-generated images and real photographs.

## Dataset Overview

- **Category:** Image Authenticity Assessment
- **Total Samples:** 6,180
- **Architecture Base:** EfficientNetV2 Feature Extractor with Distribution Scoring Head
- **Primary Task:** Predict image authenticity score and map to binary categorical distribution.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Ai Generated Vs Real** | 2,763 | 377 | 3,140 samples |
| **Real Vs Fake Faces** | 1,795 | 245 | 2,040 samples |
| **Sut Project** | 880 | 120 | 1,000 samples |

## Model Training Profiles

### Model: LemGendary Authenticity Scorer (AI vs Human)

- **Architecture**: EfficientNetV2-S (Distribution Scorer)
- **Optimization**: emd

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.77 | > 0.86 | **> 0.96** |

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
| **images** | 5,438 | 742 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizednimaauthenticitylarge)
