# LemGendary Dataset Manifolds Directory & Lineage Matrix

This document provides a comprehensive operational index, physical file structure taxonomy, sample distributions, lineage tracking, and model bindings for all 20 production dataset manifolds in the **LemGendary Dataset Suite**.

---

## 1. Master Manifolds Matrix

| Manifold Name | Domain | Category / Task | Samples | Storage Format | Upstream Sources | Bound Model(s) | Kaggle Reference |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LemGendizedForexUniverseLarge` | Financial & Time-Series | `forex` | 26,818,700 | Parquet (Annual Shards 2019-2026) | MetaTrader 5 Native Cache | `forex_predictor` | `kaggle://lemtreursi/lemgendizedforexuniverselarge` |
| `LemGendizedNimaAestheticLarge` | Image Manipulation & Restoration | `quality` | 321,369 | Directory Pair (`images/`, `labels/`) | AVA, TAD66K, KonIQ-10k, SPAQ | `nima_aesthetic_mobile`, `nima_aesthetic_efficientnet`, `nima_aesthetic_pro` | `kaggle://lemtreursi/lemgendizednimaaestheticlarge` |
| `LemGendizedNimaTechnicalLarge` | Image Manipulation & Restoration | `quality` | 26,093 | Directory Pair (`images/`, `labels/`) | KonIQ-10k, TID2013, SPAQ, LIVE, DND & NAM, CSIQ | `nima_technical` | `kaggle://lemtreursi/lemgendizednimatechnicallarge` |
| `LemGendizedNimaAuthenticityLarge` | Image Manipulation & Restoration | `authenticity` | 6,180 | Categorical Directory Distribution | AI Generated Vs Real, Real Vs Fake Faces, Sut Project | `nima_authenticity` | `kaggle://lemtreursi/lemgendizednimaauthenticitylarge` |
| `LemGendizedUpnV2Large` | Image Manipulation & Restoration | `parameter_prediction` | 1,378,070 | Paired Tensors / Parameter Tensors | DPED, Adobe FiveK, Flickr, DIV2K | `upn_v2` | `kaggle://lemtreursi/lemgendizedupnv2large` |
| `LemGendizedFilmRestorerLarge` | Image Manipulation & Restoration | `restoration` | 67,542 | Degraded / Restored Image Pairs | Vintage Film Archive, DIV2K, Vintage Photos, SIDD, DND & NAM | `film_restorer` | `kaggle://lemtreursi/lemgendizedfilmrestorerlarge` |
| `LemGendizedCodeFormerLarge` | Image Manipulation & Restoration | `restoration` | 22,000 | LQ / HQ Face Tensors | Codeformer Manifold, Synthetic Degradation Engine | `codeformer` | `kaggle://lemtreursi/lemgendizedcodeformerlarge` |
| `LemGendizedParseNetLarge` | Image Manipulation & Restoration | `segmentation` | 853,546 | Image / 19-Class Semantic Mask Pairs | SFHQ Parts 1, 2, 3, 4 | `parsenet` | `kaggle://lemtreursi/lemgendizedparsenetlarge` |
| `LemGendizedRetinaFaceMobileNetLarge` | Image Manipulation & Restoration | `detection` | 853,546 | Image / Multi-Scale Facial BBoxes & Landmarks | SFHQ Parts 1, 2, 3, 4 | `retinaface` | `kaggle://lemtreursi/lemgendizedretinafacemobilenetlarge` |
| `LemGendizedFfaNetIndoorLarge` | Image Manipulation & Restoration | `restoration` | 196,304 | Hazy / Ground-Truth Clear Pairs | RESIDE Standard Indoor, Dehazing and Desmoking | `ffanet_indoor` | `kaggle://lemtreursi/lemgendizedffanetindoorlarge` |
| `LemGendizedFfaNetOutdoorLarge` | Image Manipulation & Restoration | `restoration` | 217,113 | Hazy / Ground-Truth Clear Pairs | Outdoor Dehazing, O-HAZE, NH-HAZE, Hazing Images (CVPR) | `ffanet_outdoor` | `kaggle://lemtreursi/lemgendizedffanetoutdoorlarge` |
| `LemGendizedMirNetLowLightLarge` | Image Manipulation & Restoration | `restoration` | 15,070 | Underexposed / Corrected HDR Pairs | LOL-v2, ExDark, Low Light Datasets, SID, LOL, SIDD | `mirnet_lowlight` | `kaggle://lemtreursi/lemgendizedmirnetlowlightlarge` |
| `LemGendizedMirNetExposureLarge` | Image Manipulation & Restoration | `restoration` | 11,142 | Over/Underexposed / Normal Pairs | Adobe FiveK, DPED Smartphone Photography | `mirnet_exposure` | `kaggle://lemtreursi/lemgendizedmirnetexposurelarge` |
| `LemGendizedMprNetDerainingLarge` | Image Manipulation & Restoration | `restoration` | 34,407 | Rainy / Clean Background Pairs | Balraj Rain, Rain100L, Rain100H, High-Res Rainy Images | `mprnet_deraining` | `kaggle://lemtreursi/lemgendizedmprnetderaininglarge` |
| `LemGendizedNafNetDebluringLarge` | Image Manipulation & Restoration | `restoration` | 6,679 | Blurry / Sharp Image Pairs | GoPro Deblurring, RealBlur, HiDeBlur, Image Deblurring Performance | `nafnet_debluring` | `kaggle://lemtreursi/lemgendizednafnetdebluringlarge` |
| `LemGendizedNafNetDenoisingLarge` | Image Manipulation & Restoration | `restoration` | 1,127 | Noisy Raw / Denoised Ground Truth | SIDD, DND & NAM, Multiple Iso, 9-Classes Noisy, Multi-Noise, Salt-Pepper | `nafnet_denoising` | `kaggle://lemtreursi/lemgendizednafnetdenoisinglarge` |
| `LemGendizedUltraZoomLarge` | Image Manipulation & Restoration | `super-resolution` | 17,724 | LR / HR Sub-Pixel Image Pairs | DF2K-OST, Flickr2K, DIV2K, Urban100 | `ultrazoom` | `kaggle://lemtreursi/lemgendizedultrazoomlarge` |
| `LemGendizedYoloV8nLarge` | Image Manipulation & Restoration | `detection` | 153,972 | Images / YOLO-Format Bounding Box Text Labels | KITTI, Pascal VOC 2012, COCO 2017, CrowdPose, MPII Human Pose | `yolov8n` | `kaggle://lemtreursi/lemgendizedyolov8nlarge` |
| `LemGendizedProfessionalMultitaskRestorationLarge` | Image Manipulation & Restoration | `restoration` | 343,911 | Directory Pair (`images/`, `targets/`) | 11 Sub-Manifolds (Dehaze, Derain, Denoise, Deblur, Low-Light, Film, Zoom) | `professional_multitask_restoration` | `kaggle://lemtreursi/lemgendizedprofessionalmultitaskrestorationlarge` |
| `LemGendizedClassificationMasterManifoldLarge` | Image Generation & Multimodal | `classification` | 788,034 | Directory Pair (`images/`, `labels/`) | Anime DB Rating (Danbooru), NSFW Dataset, Food-101 | `universal_nsfw_classification` | `kaggle://lemtreursi/lemgendizedclassificationmastermanifoldlarge` |

---

## 2. Manifold Specifications & Architecture

### 2.1 LemGendizedForexUniverseLarge

- **Domain**: Financial & Time-Series
- **Task Category**: Causal Sequence Forecasting (`forex`)
- **Total Samples**: 26,818,700 temporal steps across 16 global currency and commodity symbols
- **Storage Format**: High-throughput Snappy-compressed Apache Parquet (`.parquet`), partitioned by year
- **Directory Layout**:

```text
LemGendizedForexUniverseLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── ForexUniverse2019.parquet
├── ForexUniverse2020.parquet
├── ForexUniverse2021.parquet
├── ForexUniverse2022.parquet
├── ForexUniverse2023.parquet
├── ForexUniverse2024.parquet
├── ForexUniverse2025.parquet
├── ForexUniverse2026.parquet
├── forex_predictor_training.ipynb
└── forex_predictor_colab_training.ipynb
```

- **Feature Schema**:
  - `timestamp`: Epoch millisecond causal index
  - `symbol`: Categorical asset identifier (`EURUSD`, `GBPUSD`, `USDJPY`, `XAUUSD`, etc.)
  - `timeframe`: Sampling horizon (`M1`, `M5`, `M15`, `H1`, `H4`, `D1`)
  - `open`, `high`, `low`, `close`: Normalized OHLC prices
  - `volume`, `spread`: Tick volume and broker bid-ask friction
- **Upstream Lineage**: MetaTrader 5 High-Frequency Raw Tick & Candle Server Cache.
- **Bound Model**: `forex_predictor` (Multi-Scale CNN-Transformer).

---

### 2.2 LemGendizedNimaAestheticLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Aesthetic Quality Distribution Scoring (`quality`)
- **Total Samples**: 321,369 curated photographic samples
- **Storage Format**: Indexed Directory Hierarchy with Normalized Distribution JSON
- **Directory Layout**:

```text
LemGendizedNimaAestheticLarge/
├── images/
├── labels/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── index.json
├── README.md
├── nima_aesthetic_mobile_training.ipynb
├── nima_aesthetic_mobile_colab_training.ipynb
├── nima_aesthetic_efficientnet_training.ipynb
├── nima_aesthetic_efficientnet_colab_training.ipynb
├── nima_aesthetic_pro_training.ipynb
└── nima_aesthetic_pro_colab_training.ipynb
```

- **Label Schema**: 10-bin normalized probability distribution vector corresponding to aesthetic rating scores from 1 to 10.
- **Upstream Lineage**: AVA (Aesthetic Visual Analysis), TAD66K, KonIQ-10k, and SPAQ.
- **Bound Models**: `nima_aesthetic_mobile` (MobileNetV2), `nima_aesthetic_efficientnet` (EfficientNetV2-S), `nima_aesthetic_pro` (Swin-v2-T).

---

### 2.3 LemGendizedNimaTechnicalLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Technical Distortion & Fidelity Scoring (`quality`)
- **Total Samples**: 26,093 images with graded distortion metrics
- **Storage Format**: Directory Pair (`images/`, `labels/`) with master index
- **Directory Layout**:

```text
LemGendizedNimaTechnicalLarge/
├── images/
├── labels/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── index.json
├── README.md
├── nima_technical_training.ipynb
└── nima_technical_colab_training.ipynb
```

- **Label Schema**: Continuous technical mean opinion score (MOS) and 10-bin distortion likelihood distribution.
- **Upstream Lineage**: KonIQ-10k, TID2013, SPAQ, LIVE In the Wild, DND & NAM Noise Data, CSIQ.
- **Bound Model**: `nima_technical` (EfficientNetV2-S).

---

### 2.4 LemGendizedNimaAuthenticityLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Generative vs Authentic Discrimination (`authenticity`)
- **Total Samples**: 6,180 balanced authentic and synthetic pairs
- **Storage Format**: Class-partitioned image directory structure
- **Directory Layout**:

```text
LemGendizedNimaAuthenticityLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── nima_authenticity_training.ipynb
└── nima_authenticity_colab_training.ipynb
```

- **Classes**: `authentic_camera`, `ai_generated_diffusion`.
- **Upstream Lineage**: AI Generated vs Real Benchmark, Real vs Fake Faces, Sut Project.
- **Bound Model**: `nima_authenticity` (EfficientNetV2-S).

---

### 2.5 LemGendizedUpnV2Large

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Universal Photographic Parameter Regression (`parameter_prediction`)
- **Total Samples**: 1,378,070 photographic samples with paired raw metadata
- **Storage Format**: Normalized paired image tensors and ground-truth 14-parameter adjustment vectors
- **Directory Layout**:

```text
LemGendizedUpnV2Large/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── upn_v2_training.ipynb
└── upn_v2_colab_training.ipynb
```

- **Regression Parameters**: Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Temperature, Tint, Vibrance, Saturation, Clarity, Dehaze, Vignette, Sharpness.
- **Upstream Lineage**: DPED (Smartphone Photography), Adobe FiveK, Flickr Creative Commons, DIV2K.
- **Bound Model**: `upn_v2` (MobileNet-Lite Parameter Regressor).

---

### 2.6 LemGendizedFilmRestorerLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Vintage Film and Print Artifact Removal (`restoration`)
- **Total Samples**: 67,542 severely degraded / restored ground-truth image pairs
- **Storage Format**: Paired RGB tensors with synthetic physical degradation maps
- **Directory Layout**:

```text
LemGendizedFilmRestorerLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── film_restorer_training.ipynb
└── film_restorer_colab_training.ipynb
```

- **Degradation Profiles**: Physical scratches, dust flecks, grain noise, emulsion stains, sepia fading, chromatic aberration.
- **Upstream Lineage**: Vintage & Degraded Film Archive, DIV2K, Vintage Degraded Photos, SIDD, DND & NAM.
- **Bound Model**: `film_restorer` (UniversalFilmRestorer Residual Dense Autoencoder).

---

### 2.7 LemGendizedCodeFormerLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Blind Face Restoration & Codebook Lookup (`restoration`)
- **Total Samples**: 22,000 high-fidelity facial crops with controlled degradation pipelines
- **Storage Format**: Low-Quality (LQ) / High-Quality (HQ) 512x512 aligned facial crops
- **Directory Layout**:

```text
LemGendizedCodeFormerLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── codeformer_training.ipynb
└── codeformer_colab_training.ipynb
```

- **Upstream Lineage**: Codeformer Master Manifold and Synthetic Face Degradation Engine.
- **Bound Model**: `codeformer` (Transformer-Based Vector Quantized Codebook Network).

---

### 2.8 LemGendizedParseNetLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Bilateral Face Parsing & Semantic Segmentation (`segmentation`)
- **Total Samples**: 853,546 ultra-high-resolution facial segmentation pairs
- **Storage Format**: Aligned 512x512 facial images with 19-class indexed PNG mask annotations
- **Directory Layout**:

```text
LemGendizedParseNetLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── parsenet_training.ipynb
└── parsenet_colab_training.ipynb
```

- **Classes (19)**: Background, Skin, Left Eyebrow, Right Eyebrow, Left Eye, Right Eye, Nose, Upper Lip, Inner Mouth, Lower Lip, Hair, Left Ear, Right Ear, Eyeglasses, Earring, Necklace, Neck, Cloth, Hat.
- **Upstream Lineage**: SFHQ (Synthetic Faces High Quality) Parts 1, 2, 3, and 4.
- **Bound Model**: `parsenet` (Bilateral Face Segmentation Network).

---

### 2.9 LemGendizedRetinaFaceMobileNetLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Multi-Scale Single-Shot Face & Landmark Localization (`detection`)
- **Total Samples**: 853,546 annotated facial images
- **Storage Format**: Image tensors paired with normalized bounding boxes and 5-point facial landmark coordinates (left eye, right eye, nose, left mouth corner, right mouth corner)
- **Directory Layout**:

```text
LemGendizedRetinaFaceMobileNetLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── retinaface_training.ipynb
└── retinaface_colab_training.ipynb
```

- **Upstream Lineage**: SFHQ (Synthetic Faces High Quality) Parts 1, 2, 3, and 4.
- **Bound Model**: `retinaface` (MobileNetV1-0.25 FPN Feature Pyramid Network).

---

### 2.10 LemGendizedFfaNetIndoorLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Indoor Atmospheric Dehazing (`restoration`)
- **Total Samples**: 196,304 synthetic and physical indoor hazy/clear pairs
- **Storage Format**: Paired RGB images (Scattered Haze Input / Ground Truth Transmission-Cleared Output)
- **Directory Layout**:

```text
LemGendizedFfaNetIndoorLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── ffanet_indoor_training.ipynb
└── ffanet_indoor_colab_training.ipynb
```

- **Upstream Lineage**: RESIDE Standard Indoor Dataset, Dehazing and Desmoking Master Archive.
- **Bound Model**: `ffanet_indoor` (BranchedFFANet Indoor Configuration).

---

### 2.11 LemGendizedFfaNetOutdoorLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Outdoor Landscape Atmospheric Dehazing (`restoration`)
- **Total Samples**: 217,113 outdoor atmospheric haze pairs
- **Storage Format**: Paired RGB images with non-uniform dense haze distributions
- **Directory Layout**:

```text
LemGendizedFfaNetOutdoorLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── ffanet_outdoor_training.ipynb
└── ffanet_outdoor_colab_training.ipynb
```

- **Upstream Lineage**: Outdoor Dehazing Dataset, O-HAZE / NTIRE Dehazing, NH-HAZE Dataset, Hazing Images Dataset (CVPR).
- **Bound Model**: `ffanet_outdoor` (BranchedFFANet Outdoor Configuration).

---

### 2.12 LemGendizedMirNetLowLightLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Extreme Low-Light Enhancement & Denoising (`restoration`)
- **Total Samples**: 15,070 severely underexposed and photon-starved image pairs
- **Storage Format**: Low-Light RAW/RGB inputs paired with long-exposure ground-truth references
- **Directory Layout**:

```text
LemGendizedMirNetLowLightLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── mirnet_lowlight_training.ipynb
└── mirnet_lowlight_colab_training.ipynb
```

- **Upstream Lineage**: LOL-v2 Dataset, ExDark (Exclusively Dark), Low Light Image Enhancement Datasets, Learning to See in the Dark (SID), LOL Dataset, SIDD.
- **Bound Model**: `mirnet_lowlight` (MIRNet_v2 Low-Light Mode).

---

### 2.13 LemGendizedMirNetExposureLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Multi-Exposure Correction & Tone Mapping (`restoration`)
- **Total Samples**: 11,142 non-uniformly exposed photographs
- **Storage Format**: Overexposed and underexposed inputs paired with expert-retouched balanced targets
- **Directory Layout**:

```text
LemGendizedMirNetExposureLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── mirnet_exposure_training.ipynb
└── mirnet_exposure_colab_training.ipynb
```

- **Upstream Lineage**: Adobe FiveK, DPED (Smartphone Photography).
- **Bound Model**: `mirnet_exposure` (MIRNet_v2 Exposure Mode).

---

### 2.14 LemGendizedMprNetDerainingLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Multi-Stage Rain Streak & Droplet Removal (`restoration`)
- **Total Samples**: 34,407 synthetic and real-world rain scenes
- **Storage Format**: Rainy input images paired with clean clear background references
- **Directory Layout**:

```text
LemGendizedMprNetDerainingLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── mprnet_deraining_training.ipynb
└── mprnet_deraining_colab_training.ipynb
```

- **Upstream Lineage**: Balraj Rain Dataset, Rain100L, Rain100H, High-Resolution Rainy Images.
- **Bound Model**: `mprnet_deraining` (MPRNet Multi-Stage Progressive Restorer).

---

### 2.15 LemGendizedNafNetDebluringLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Motion & Defocus Blur Elimination (`restoration`)
- **Total Samples**: 6,679 dynamic camera motion and optical blur pairs
- **Storage Format**: High-speed camera blurry frames paired with crystal-sharp references
- **Directory Layout**:

```text
LemGendizedNafNetDebluringLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── nafnet_debluring_training.ipynb
└── nafnet_debluring_colab_training.ipynb
```

- **Upstream Lineage**: GoPro Deblurring Dataset, RealBlur Dataset, HiDeBlur Dataset, Image Deblurring Performance Benchmark.
- **Bound Model**: `nafnet_debluring` (NAFNet Deblurring Mode).

---

### 2.16 LemGendizedNafNetDenoisingLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Sensor ISO Noise & Grain Suppression (`restoration`)
- **Total Samples**: 1,127 multi-ISO calibrated noisy/clean ground-truth image pairs
- **Storage Format**: RAW sensor crops and sRGB pairs across wide ISO ranges (100 to 25600)
- **Directory Layout**:

```text
LemGendizedNafNetDenoisingLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── nafnet_denoising_training.ipynb
└── nafnet_denoising_colab_training.ipynb
```

- **Upstream Lineage**: SIDD (Smartphone Image Denoising), DND & NAM Noise Data, Multiple ISO Denoising Dataset, 9-Classes Noisy Image Dataset, Multi-Noise Synthetic Dataset, Salt-and-Pepper Noise.
- **Bound Model**: `nafnet_denoising` (NAFNet Denoising Mode).

---

### 2.17 LemGendizedUltraZoomLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Sub-Pixel ESPCN 4x Super-Resolution (`super-resolution`)
- **Total Samples**: 17,724 high-frequency photographic crops
- **Storage Format**: Bicubic / real-world low-resolution inputs paired with 4x high-resolution targets
- **Directory Layout**:

```text
LemGendizedUltraZoomLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── ultrazoom_training.ipynb
└── ultrazoom_colab_training.ipynb
```

- **Upstream Lineage**: DF2K-OST, Flickr2K, DIV2K Dataset, Urban100.
- **Bound Model**: `ultrazoom` (UltraZoomMaster Sub-Pixel Architecture).

---

### 2.18 LemGendizedYoloV8nLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Real-Time Multi-Class Object & Human Pose Detection (`detection`)
- **Total Samples**: 153,972 annotated images
- **Storage Format**: Standard YOLO format text labels with normalized coordinates `[class_id x_center y_center width height]`
- **Directory Layout**:

```text
LemGendizedYoloV8nLarge/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── README.md
├── yolov8n_training.ipynb
└── yolov8n_colab_training.ipynb
```

- **Upstream Lineage**: KITTI Vision Benchmark, Pascal VOC 2012, COCO 2017, CrowdPose Dataset, MPII Human Pose.
- **Bound Model**: `yolov8n` (CSPDarknet53 + PANet Detection Engine).

---

### 2.19 LemGendizedProfessionalMultitaskRestorationLarge

- **Domain**: Image Manipulation & Restoration
- **Task Category**: Comprehensive Multi-Task Restoration Mixture-of-Experts (`restoration`)
- **Total Samples**: 343,911 multi-degradation samples
- **Storage Format**: Unified directory pair (`images/`, `targets/`) indexed via `index.json`
- **Directory Layout**:

```text
LemGendizedProfessionalMultitaskRestorationLarge/
├── images/
├── targets/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── index.json
├── README.md
├── professional_multitask_restoration_training.ipynb
└── professional_multitask_restoration_colab_training.ipynb
```

- **Task Modes**: Dehazing, Deraining, Denoising, Deblurring, Low-Light Recovery, Exposure Balancing, Film Scratch Elimination, Super-Resolution.
- **Upstream Lineage**: Aggregated across 11 specialized sub-manifolds (FFANet Indoor/Outdoor, Balraj Rain, ParseNet, CodeFormer, NAFNet Denoise/Deblur, MIRNet Exposure/Low-Light, UltraZoom, Film Restorer).
- **Bound Model**: `professional_multitask_restoration` (MultiTaskRestorer Shared Encoder MoE).

---

### 2.20 LemGendizedClassificationMasterManifoldLarge

- **Domain**: Image Generation & Multimodal / Multi-Domain
- **Task Category**: Multi-Class Categorical Safety & NSFW Content Moderation (`classification`)
- **Total Samples**: 788,034 labeled photographic and illustrative samples
- **Storage Format**: Directory Pair (`images/`, `labels/`) indexed via `index.json`
- **Directory Layout**:

```text
LemGendizedClassificationMasterManifoldLarge/
├── images/
├── labels/
├── category.txt
├── classes.txt
├── dataset_info.yaml
├── index.json
├── README.md
├── universal_nsfw_classification_training.ipynb
└── universal_nsfw_classification_colab_training.ipynb
```

- **Classification Categories**: Safe For Work (SFW), Suggestive, Explicit NSFW, Anime / Illustration, Real Photograph.
- **Upstream Lineage**: Anime DB Rating (Danbooru), NSFW Image Dataset, Food-101 Benchmark.
- **Bound Model**: `universal_nsfw_classification` (EfficientNetV2-S Multi-Class Classifier).

## 3 Format Choice & Conversion Utilities

Yes. While WebDataset (.tar) and LMDB are strong traditional upgrades over raw directories, they have notable architectural limitations. WebDataset struggles with true random shuffling and mid-epoch training resumption. LMDB databases can suffer from inflated file sizes because they lack modern deep-learning-native compression pipelines. [1, 2]
The industry has evolved to use Streaming-Native and Hardware-Accelerated file formats specifically designed to drop dataset footprints while maximizing GPU utilization. [3, 4]
The three primary modern alternatives that will reduce your storage footprint and improve training performance are detailed below.

## 3 Modern Re-Architected Strategy Matrix

Implementing these modern additions updates your dataset configuration as follows:

| Modern Target Format | Targeted Datasets | Practical Improvement Realized |
| --- | --- | --- |
| MosaicML MDS (.mds) | UpnV2, ParseNet, RetinaFaceMobileNet, ClassificationMasterManifold, ProfessionalMultitaskRestoration, NimaAesthetic | Saves up to 40% disk space via Zstd compression. Grants instant mid-epoch crash resilience and true global shuffling across multi-GPU setups. |
| FFCV (.beton) | FilmRestorer, FfaNetIndoor, FfaNetOutdoor, MirNetLowLight, MirNetExposure, MprNetDeraining, NafNetDebluring | Eliminated CPU bottlenecking. Compiles data streams directly to machine code, maximizing the compute saturation of the Dual T4 accelerators. |
| LitData | YoloV8n | Streamlined serialization of localized string arrays alongside target image frames without filesystem strain. |
| Optimized Parquet | ForexUniverse | Optimized tabular structure allowing vector filtering right at the storage level. |
