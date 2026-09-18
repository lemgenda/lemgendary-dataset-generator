# LemGendizedParseNetLarge

> Detailed dataset for semantic and instance segmentation.

## Dataset Overview

- **Category:** Image Segmentation
- **Total Samples:** 853,546
- **Architecture Base:** Bilateral Segmentation Network / DeepLabV3+ with ResNet Backbone
- **Primary Task:** Assign categorical labels to every pixel in the image manifold.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **SFHQ (Synthetic Faces High Quality) Part 4** | 222,166 | 30,122 | 252,288 samples |
| **SFHQ (Synthetic Faces High Quality) Part 3** | 208,930 | 28,666 | 237,596 samples |
| **SFHQ (Synthetic Faces High Quality) Part 2** | 161,388 | 22,124 | 183,512 samples |
| **SFHQ (Synthetic Faces High Quality) Part 1** | 158,499 | 21,651 | 180,150 samples |

## Model Training Profiles

### Model: LemGendary ParseNet Face Parsing

- **Architecture**: ParseNet (Bilateral Face Segmentation Network)
- **Optimization**: cross_entropy

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **mIoU** | ~0.69 | > 0.77 | **> 0.86** |

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
| **images** | 750,983 | 102,563 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedparsenetlarge)
