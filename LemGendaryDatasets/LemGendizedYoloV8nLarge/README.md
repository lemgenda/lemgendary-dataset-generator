# LemGendizedYoloV8nLarge

> Industrial dataset for object detection and localization.

## Dataset Overview

- **Category:** Object Detection
- **Total Samples:** 153,972
- **Architecture Base:** Path Aggregation Network (PANet) with Darknet Backbone
- **Primary Task:** Detect and localize multiple object classes with high precision.

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
| **COCO 2017** | 35,173 | 4,743 | 39,916 samples |
| **Pascal VOC 2012** | 34,380 | 4,694 | 39,074 samples |
| **KITTI Vision Benchmark** | 26,431 | 3,567 | 29,998 samples |
| **MPII Human Pose** | 22,014 | 2,970 | 24,984 samples |
| **CrowdPose Dataset** | 17,661 | 2,339 | 20,000 samples |

## Model Training Profiles

### Model: LemGendary YOLOv8n Multi-Task Model

- **Architecture**: YOLOv8n (CSPDarknet53 + PANet)
- **Optimization**: yolo

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **mAP50** | ~0.43 | > 0.49 | **> 0.54** |
| **mAP50 95** | ~0.31 | > 0.35 | **> 0.39** |

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
| **images** | 135,659 | 18,313 |
| **labels** | 135,659 | 18,313 |

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedyolov8nlarge)
