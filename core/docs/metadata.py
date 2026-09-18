"""LemGendary Dataset Documentation Metadata.

Defines manifold mappings, column specifications, task architecture archetypes,
and source name normalization dictionaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_META_FILE = ROOT / "models" / "models_metadata.yaml"
if not MODELS_META_FILE.exists():
    MODELS_META_FILE = ROOT / "models_metadata.yaml"

_meta_data: dict[str, Any] = {}
if MODELS_META_FILE.exists():
    with open(MODELS_META_FILE, "r", encoding="utf-8") as f:
        _meta_data = yaml.safe_load(f) or {}

UNIFIED_DATA_FILE = ROOT / "unified_data.yaml"
UNIFIED_DATA: dict[str, Any] = {}
if UNIFIED_DATA_FILE.exists():
    with open(UNIFIED_DATA_FILE, "r", encoding="utf-8") as f:
        UNIFIED_DATA = yaml.safe_load(f) or {}

TASK_META: dict[str, Any] = _meta_data.get("task_metadata", {})
MODELS_META: dict[str, Any] = _meta_data.get("models_metadata", {})

MANIFOLD_TASK_MAP: dict[str, str] = {
    # Modern (2026 suffix-free)
    "LemGendizedForexUniverse": "forex",
    "LemGendizedClassificationMasterManifold": "classification",
    "LemGendizedNimaAesthetic": "quality",
    "LemGendizedNimaTechnical": "quality",
    "LemGendizedNimaAuthenticity": "authenticity",
    "LemGendizedUpnV2": "parameter_prediction",
    "LemGendizedFilmRestorer": "restoration",
    "LemGendizedCodeFormer": "restoration",
    "LemGendizedParseNet": "segmentation",
    "LemGendizedRetinaFaceMobileNet": "detection",
    "LemGendizedFfaNetIndoor": "restoration",
    "LemGendizedFfaNetOutdoor": "restoration",
    "LemGendizedMirNetLowLight": "restoration",
    "LemGendizedMirNetExposure": "restoration",
    "LemGendizedMprNetDeraining": "restoration",
    "LemGendizedNafNetDebluring": "restoration",
    "LemGendizedNafNetDenoising": "restoration",
    "LemGendizedUltraZoom": "super-resolution",
    "LemGendizedYoloV8n": "detection",
    "LemGendizedProfessionalMultitaskRestoration": "restoration",
    # Legacy (with 'Large' suffix)
    "LemGendizedForexUniverseLarge": "forex",
    "LemGendizedClassificationMasterManifoldLarge": "classification",
    "LemGendizedNimaAestheticLarge": "quality",
    "LemGendizedNimaTechnicalLarge": "quality",
    "LemGendizedNimaAuthenticityLarge": "authenticity",
    "LemGendizedUpnV2Large": "parameter_prediction",
    "LemGendizedFilmRestorerLarge": "restoration",
    "LemGendizedCodeFormerLarge": "restoration",
    "LemGendizedParseNetLarge": "segmentation",
    "LemGendizedRetinaFaceMobileNetLarge": "detection",
    "LemGendizedFfaNetIndoorLarge": "restoration",
    "LemGendizedFfaNetOutdoorLarge": "restoration",
    "LemGendizedMirNetLowLightLarge": "restoration",
    "LemGendizedMirNetExposureLarge": "restoration",
    "LemGendizedMprNetDerainingLarge": "restoration",
    "LemGendizedNafNetDebluringLarge": "restoration",
    "LemGendizedNafNetDenoisingLarge": "restoration",
    "LemGendizedUltraZoomLarge": "super-resolution",
    "LemGendizedYoloV8nLarge": "detection",
    "LemGendizedProfessionalMultitaskRestorationLarge": "restoration",
}

FOREX_COLUMN_FIELDS: list[dict[str, str]] = [
    {
        "name": "pair",
        "type": "string",
        "description": "Asset / currency pair / commodity symbol identifier (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD, NAS100, DE40, USOIL, US500).",
    },
    {
        "name": "timeframe",
        "type": "integer",
        "description": "Bar aggregation timeframe rung in minutes: 1=M1 (1min), 5=M5 (5min), 15=M15 (15min), 60=H1 (60min), 240=H4 (240min), 1440=D1 (1440min).",
    },
    {
        "name": "timestamp",
        "type": "integer",
        "description": "Millisecond Unix epoch timestamp of the sequence prediction anchor / candle close.",
    },
    {
        "name": "y_dir",
        "type": "integer",
        "description": "Causal directional classification target label over forward horizon: 0=SELL (Down), 1=HOLD (Sideways/Neutral), 2=BUY (Up).",
    },
    {
        "name": "tp_pips",
        "type": "number",
        "description": "Optimal forward Take-Profit target excursion magnitude in pips.",
    },
    {
        "name": "sl_pips",
        "type": "number",
        "description": "Maximum adverse excursion Stop-Loss safety threshold in pips.",
    },
    {
        "name": "seq_len",
        "type": "integer",
        "description": "Historical lookback sequence length in bars (e.g., 168 for H1 macro, 512 for M1 microstructure).",
    },
    {
        "name": "n_features",
        "type": "integer",
        "description": "Number of input feature dimensions per timestep (14 channels: OHLCV, RSI, MACD, MACD Signal, ATR, Bollinger Band Width, Session Sin/Cos, ATR Percentile, Bar Range Ratio).",
    },
    {
        "name": "features",
        "type": "bytes",
        "description": "Serialized float32 binary tensor representing the normalized [seq_len, n_features] temporal feature matrix.",
    },
]

MANIFEST_CACHE_PATH = ROOT / ".cache" / "manifest_cache.json"
if not MANIFEST_CACHE_PATH.exists():
    MANIFEST_CACHE_PATH = ROOT / "manifest_cache.json"

MANIFEST_CACHE: dict[str, Any] = {}
if MANIFEST_CACHE_PATH.exists():
    try:
        with open(MANIFEST_CACHE_PATH, "r", encoding="utf-8") as f:
            MANIFEST_CACHE = json.load(f)
    except Exception:
        MANIFEST_CACHE = {}

TASK_ARCH_BASE: dict[str, str] = {
    "super-resolution": "Transformer-based or Deep Residual networks",
    "pose": "Feature Pyramid Network (FPN) with MobileNet Backbone",
    "detection": "Path Aggregation Network (PANet) with Darknet Backbone",
    "segmentation": "Bilateral Segmentation Network / DeepLabV3+ with ResNet Backbone",
    "restoration": "Multi-Scale Progressive Restoration / Nonlinear Activation-Free Network",
    "quality": "Deep Convolutional Network / Vision Transformer with Earth Mover's Distance Optimization",
    "authenticity": "EfficientNetV2 Feature Extractor with Distribution Scoring Head",
    "classification": "MobileNetV2 / EfficientNet Categorical Embedding Network",
    "parameter_prediction": "Deep Multi-Layer Perceptron / Convolutional Regressor",
    "diffusion": "Latent Diffusion Model with UNet / Transformer Backbone",
    "forex": "Multi-Scale CNN-Transformer (Causal TCN + Cross-Timeframe Attention)",
}


def format_source(name: str) -> str:
    """Format and normalize raw source dataset names into canonical display titles."""
    lower_name = name.lower()
    exact_map = {
        "celebamask": "CelebAMask",
        "affectnet": "AffectNet",
        "wflw": "WFLW",
        "helen": "Helen",
        "tid2013": "TID2013",
        "laion": "LAION",
        "laion-5b": "LAION-5B",
        "df2k": "DF2K-OST",
        "flickr2k": "Flickr2K",
        "div2k": "DIV2K Dataset",
        "urban100": "Urban100",
        "coco": "COCO 2017",
        "rain100h": "Rain100H",
        "rain100l": "Rain100L",
        "nsfw": "NSFW Dataset",
        "food101": "Food-101",
        "tad66k": "TAD66K Aesthetics",
        "adobe": "Adobe FiveK",
        "dped": "DPED (Smartphone Photography)",
    }
    if lower_name in exact_map:
        return exact_map[lower_name]
    if lower_name in ["ava", "aadb", "coco", "csiq", "spaq", "live"]:
        return lower_name.upper()

    if "synthetic-faces" in lower_name or "sfhq" in lower_name:
        parts = lower_name.split("-")
        suffix = f" Part {parts[-1]}" if parts[-1].isdigit() else ""
        return f"SFHQ (Synthetic Faces High Quality){suffix}"

    if lower_name.startswith("compiled_"):
        clean = lower_name.replace("compiled_", "").replace("multitask", "").replace("MultiTask", "")
        return f"{clean} Multi-Task Sub-Manifold"

    patterns: list[tuple[list[str], str]] = [
        (["ffhq", "flickr-faces-hq"], "FFHQ (Flickr-Faces-HQ)"),
        (["koniq10k", "koniq"], "KonIQ-10k"),
        (["smartphone", "sidd"], "SIDD (Smartphone Image Denoising)"),
        (["dnd", "nam"], "DND & NAM Noise Data"),
        (["9-classes"], "9-Classes Noisy Image Dataset"),
        (["multi-noises"], "Multi-Noise Synthetic Dataset"),
        (["salt-and-pepper"], "Salt-and-Pepper Noise"),
        (["iso-levels"], "Multiple ISO Denoising Dataset"),
        (["gopro"], "GoPro Deblurring Dataset"),
        (["hideblur"], "HiDeBlur Dataset"),
        (["realblur"], "RealBlur Dataset"),
        (["coco-2017"], "COCO 2017"),
        (["pascal", "voc"], "Pascal VOC 2012"),
        (["kitti"], "KITTI Vision Benchmark"),
        (["crowdpose"], "CrowdPose Dataset"),
        (["mpii"], "MPII Human Pose"),
        (["reside", "indoor-training-set"], "RESIDE Standard Indoor"),
        (["dehazing-and-desmoking"], "Dehazing and Desmoking"),
        (["outdoor-dehazing"], "Outdoor Dehazing Dataset"),
        (["ohaze", "ntire"], "O-HAZE / NTIRE Dehazing"),
        (["nhhaze"], "NH-HAZE Dataset"),
        (["hazing-images"], "Hazing Images Dataset (CVPR)"),
        (["lol-v2"], "LOL-v2 Dataset"),
        (["lol"], "LOL (Low-Light) Dataset"),
        (["exdark"], "ExDark (Exclusively Dark)"),
        (["learning-to-see-in-the-dark", "sid"], "Learning to See in the Dark (SID)"),
        (["anime_dbrating"], "Anime DB Rating (Danbooru)"),
        (["high-resolution", "high_resolution"], "High-Resolution Rainy Images"),
        (["rain-dataset", "rain_dataset", "rain dataset"], "Balraj Rain Dataset"),
        (["rain"], "Rain Streaks Dataset"),
        (["vintage"], "Vintage Degraded Photos"),
        (["old-photo", "old_photo", "old-film", "old"], "Vintage & Degraded Film Archive"),
        (["photo-restoration", "photo"], "Photo Restoration Dataset"),
        (["realvsfake", "real_vs_fake"], "Real vs Fake Faces"),
        (["sut-project"], "SUT Project Authenticity"),
        (["ai-generated"], "AI Generated vs Real Images"),
    ]
    for keys, title in patterns:
        if any(k in lower_name for k in keys):
            return title

    return name.replace("-", " ").replace("_", " ").title()
