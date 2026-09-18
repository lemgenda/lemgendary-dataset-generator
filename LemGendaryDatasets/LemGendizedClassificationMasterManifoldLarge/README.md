# LemGendizedClassificationMasterManifoldLarge

> High-throughput dataset for image classification and content tagging.

## Dataset Overview

- **Category:** Image Classification
- **Total Samples:** 788,034
- **Architecture Base:** MobileNetV2 / EfficientNet Categorical Embedding Network
- **Primary Task:** Predict categorical classes and safety content labels.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **Anime DB Rating (Danbooru)** | 544,611 | 74,903 | 619,514 samples |
| **Food-101** | 89,105 | 11,894 | 101,000 samples |
| **NSFW Dataset** | 59,324 | 8,168 | 67,520 samples |

## Model Training Profiles

### Model: LemGendary Universal NSFW Classifier

- **Architecture**: EfficientNetV2-S (Multi-Class Categorical Head)
- **Optimization**: cross_entropy

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Accuracy** | ~0.78 | > 0.88 | **> 0.98** |

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
| **images** | 693,040 | 94,965 |
| **labels** | 693,040 | 94,965 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedclassificationmastermanifoldlarge)
