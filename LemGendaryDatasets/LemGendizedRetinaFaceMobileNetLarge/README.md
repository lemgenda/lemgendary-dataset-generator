# LemGendizedRetinaFaceMobileNetLarge

> Industrial dataset for object detection and localization.

## Dataset Overview

- **Category:** Object Detection
- **Total Samples:** 853,546
- **Architecture Base:** Path Aggregation Network (PANet) with Darknet Backbone
- **Primary Task:** Detect and localize multiple object classes with high precision.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **SFHQ (Synthetic Faces High Quality) Part 4** | 221,930 | 30,358 | 252,288 samples |
| **SFHQ (Synthetic Faces High Quality) Part 3** | 209,119 | 28,477 | 237,596 samples |
| **SFHQ (Synthetic Faces High Quality) Part 2** | 161,509 | 22,003 | 183,512 samples |
| **SFHQ (Synthetic Faces High Quality) Part 1** | 158,595 | 21,555 | 180,150 samples |

## Model Training Profiles

### Model: LemGendary RetinaFace Detection

- **Architecture**: RetinaFace (MobileNetV1-0.25 FPN Backbone)
- **Optimization**: retinaface

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **mAP Easy** | ~0.73 | > 0.82 | **> 0.915** |
| **mAP Medium** | ~0.71 | > 0.80 | **> 0.89** |
| **mAP Hard** | ~0.60 | > 0.68 | **> 0.75** |

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
| **images** | 751,153 | 102,393 |
| **labels** | 751,153 | 102,393 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedretinafacemobilenetlarge)
