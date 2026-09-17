"""
LemGendary Dataset Documentation Generator
Generates index.json, dataset_info.yaml, category.txt, classes.txt, and README.md.
"""

import json
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
import re

# ─── Load metadata ──────────────────────────────────────────────────────
from typing import Any

MODELS_META_FILE = Path(__file__).parent / "models_metadata.yaml"
_meta_data: dict[str, Any] = {}
if MODELS_META_FILE.exists():
    with open(MODELS_META_FILE, "r", encoding="utf-8") as f:
        _meta_data = yaml.safe_load(f) or {}

UNIFIED_DATA_FILE = Path(__file__).parent / "unified_data.yaml"
UNIFIED_DATA: dict[str, Any] = {}
if UNIFIED_DATA_FILE.exists():
    with open(UNIFIED_DATA_FILE, "r", encoding="utf-8") as f:
        UNIFIED_DATA = yaml.safe_load(f) or {}

TASK_META: dict[str, Any] = _meta_data.get("task_metadata", {})
MODELS_META: dict[str, Any] = _meta_data.get("models_metadata", {})

MANIFOLD_TASK_MAP = {
    # ── Modern (2026 suffix-free) — added Phase 0 ──────────────────────────
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
    # ── Legacy (with 'Large' suffix) — kept for backward compatibility ─────
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

FOREX_COLUMN_FIELDS = [
    {
        "name": "pair",
        "type": "string",
        "description": "Asset / currency pair / commodity symbol identifier (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD, NAS100, DE40, USOIL, US500)."
    },
    {
        "name": "timeframe",
        "type": "integer",
        "description": "Bar aggregation timeframe rung in minutes: 1=M1 (1min), 5=M5 (5min), 15=M15 (15min), 60=H1 (60min), 240=H4 (240min), 1440=D1 (1440min)."
    },
    {
        "name": "timestamp",
        "type": "integer",
        "description": "Millisecond Unix epoch timestamp of the sequence prediction anchor / candle close."
    },
    {
        "name": "y_dir",
        "type": "integer",
        "description": "Causal directional classification target label over forward horizon: 0=SELL (Down), 1=HOLD (Sideways/Neutral), 2=BUY (Up)."
    },
    {
        "name": "tp_pips",
        "type": "number",
        "description": "Optimal forward Take-Profit target excursion magnitude in pips."
    },
    {
        "name": "sl_pips",
        "type": "number",
        "description": "Maximum adverse excursion Stop-Loss safety threshold in pips."
    },
    {
        "name": "seq_len",
        "type": "integer",
        "description": "Historical lookback sequence length in bars (e.g., 168 for H1 macro, 512 for M1 microstructure)."
    },
    {
        "name": "n_features",
        "type": "integer",
        "description": "Number of input feature dimensions per timestep (14 channels: OHLCV, RSI, MACD, MACD Signal, ATR, Bollinger Band Width, Session Sin/Cos, ATR Percentile, Bar Range Ratio)."
    },
    {
        "name": "features",
        "type": "bytes",
        "description": "Serialized float32 binary tensor representing the normalized [seq_len, n_features] temporal feature matrix."
    }
]


# ─── Manifest cache ──────────────────────────────────────────────────
MANIFEST_CACHE_PATH = Path(__file__).parent / "manifest_cache.json"
MANIFEST_CACHE = {}
if MANIFEST_CACHE_PATH.exists():
    try:
        with open(MANIFEST_CACHE_PATH, "r", encoding="utf-8") as f:
            MANIFEST_CACHE = json.load(f)
    except Exception:
        MANIFEST_CACHE = {}

TASK_ARCH_BASE = {
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
    "forex": "Multi-Scale CNN-Transformer (Causal TCN + Cross-Timeframe Attention)"
}


def format_source(name):
    lower_name = name.lower()
    exact_map = {
        'celebamask': 'CelebAMask',
        'affectnet': 'AffectNet',
        'wflw': 'WFLW',
        'helen': 'Helen',
        'tid2013': 'TID2013',
        'laion': 'LAION',
        'laion-5b': 'LAION-5B',
        'df2k': 'DF2K-OST',
        'flickr2k': 'Flickr2K',
        'div2k': 'DIV2K Dataset',
        'urban100': 'Urban100',
        'coco': 'COCO 2017',
        'rain100h': 'Rain100H',
        'rain100l': 'Rain100L',
        'nsfw': 'NSFW Dataset',
        'food101': 'Food-101',
        'tad66k': 'TAD66K Aesthetics',
        'adobe': 'Adobe FiveK',
        'dped': 'DPED (Smartphone Photography)',
    }
    if lower_name in exact_map:
        return exact_map[lower_name]
    if lower_name in ['ava', 'aadb', 'coco', 'csiq', 'spaq', 'live']:
        return lower_name.upper()

    if 'synthetic-faces' in lower_name or 'sfhq' in lower_name:
        parts = lower_name.split('-')
        suffix = f" Part {parts[-1]}" if parts[-1].isdigit() else ""
        return f"SFHQ (Synthetic Faces High Quality){suffix}"

    if lower_name.startswith('compiled_'):
        clean = lower_name.replace('compiled_', '').replace('multitask', '').replace('MultiTask', '')
        return f"{clean} Multi-Task Sub-Manifold"

    patterns = [
        (['ffhq', 'flickr-faces-hq'], 'FFHQ (Flickr-Faces-HQ)'),
        (['koniq10k', 'koniq'], 'KonIQ-10k'),
        (['smartphone', 'sidd'], 'SIDD (Smartphone Image Denoising)'),
        (['dnd', 'nam'], 'DND & NAM Noise Data'),
        (['9-classes'], '9-Classes Noisy Image Dataset'),
        (['multi-noises'], 'Multi-Noise Synthetic Dataset'),
        (['salt-and-pepper'], 'Salt-and-Pepper Noise'),
        (['iso-levels'], 'Multiple ISO Denoising Dataset'),
        (['gopro'], 'GoPro Deblurring Dataset'),
        (['hideblur'], 'HiDeBlur Dataset'),
        (['realblur'], 'RealBlur Dataset'),
        (['coco-2017'], 'COCO 2017'),
        (['pascal', 'voc'], 'Pascal VOC 2012'),
        (['kitti'], 'KITTI Vision Benchmark'),
        (['crowdpose'], 'CrowdPose Dataset'),
        (['mpii'], 'MPII Human Pose'),
        (['reside', 'indoor-training-set'], 'RESIDE Standard Indoor'),
        (['dehazing-and-desmoking'], 'Dehazing and Desmoking'),
        (['outdoor-dehazing'], 'Outdoor Dehazing Dataset'),
        (['ohaze', 'ntire'], 'O-HAZE / NTIRE Dehazing'),
        (['nhhaze'], 'NH-HAZE Dataset'),
        (['hazing-images'], 'Hazing Images Dataset (CVPR)'),
        (['lol-v2'], 'LOL-v2 Dataset'),
        (['lol'], 'LOL (Low-Light) Dataset'),
        (['exdark'], 'ExDark (Exclusively Dark)'),
        (['learning-to-see-in-the-dark', 'sid'], 'Learning to See in the Dark (SID)'),
        (['anime_dbrating'], 'Anime DB Rating (Danbooru)'),
        (['high-resolution', 'high_resolution'], 'High-Resolution Rainy Images'),
        (['rain-dataset', 'rain_dataset', 'rain dataset'], 'Balraj Rain Dataset'),
        (['rain'], 'Rain Streaks Dataset'),
        (['vintage'], 'Vintage Degraded Photos'),
        (['old-photo', 'old_photo', 'old-film', 'old'], 'Vintage & Degraded Film Archive'),
        (['photo-restoration', 'photo'], 'Photo Restoration Dataset'),
        (['realvsfake', 'real_vs_fake'], 'Real vs Fake Faces'),
        (['sut-project'], 'SUT Project Authenticity'),
        (['ai-generated'], 'AI Generated vs Real Images'),
    ]
    for keys, title in patterns:
        if any(k in lower_name for k in keys):
            return title

    return name.replace('-', ' ').replace('_', ' ').title()


# ─── Forex scanner ──────────────────────────────────────────────────────
def scan_forex_manifold(root_path):
    result = {
        "years": [],
        "pairs": [],
        "timeframes": [],
        "samples_per_year": {},
        "samples_per_pair": {},
        "samples_per_tf": {},
        "details": []
    }

    year_set = set()
    pair_set = set()
    tf_set = set()

    # 1. Scan unified Parquet files first (ForexUniverseYYYY.parquet)
    for pq_file in sorted(root_path.glob("ForexUniverse*.parquet")):
        year_str = pq_file.stem.replace("ForexUniverse", "")
        try:
            year = int(year_str)
        except ValueError:
            continue
        year_set.add(year)

        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(pq_file))
            tbl = pf.read(columns=["pair", "timeframe"])
            p_arr = tbl["pair"].to_numpy(zero_copy_only=False)
            tf_arr = tbl["timeframe"].to_numpy()

            # Count occurrences of (pair, tf)
            from collections import Counter
            counts = Counter(zip(p_arr, tf_arr))
            for (pair, tf), count in counts.items():
                pair_set.add(pair)
                tf_set.add(int(tf))
                entry = {
                    "year": year,
                    "pair": pair,
                    "timeframe": int(tf),
                    "count": count
                }
                result["details"].append(entry)
                result["samples_per_year"][year] = result["samples_per_year"].get(year, 0) + count
                result["samples_per_pair"][pair] = result["samples_per_pair"].get(pair, 0) + count
                result["samples_per_tf"][int(tf)] = result["samples_per_tf"].get(int(tf), 0) + count
        except Exception as e:
            print(f" [WARNING] Error scanning {pq_file.name}: {e}")

    # 2. Scan legacy directories if not already scanned as Parquet
    for chunk_dir in root_path.glob("ForexUniverse*"):
        if not chunk_dir.is_dir():
            continue
        year_str = chunk_dir.name.replace("ForexUniverse", "")
        try:
            year = int(year_str)
        except ValueError:
            continue

        if year in year_set:
            continue
        year_set.add(year)

        for pair_dir in chunk_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            pair = pair_dir.name
            pair_set.add(pair)

            for tf_dir in pair_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                tf_str = tf_dir.name
                try:
                    tf = int(tf_str)
                except ValueError:
                    continue
                tf_set.add(tf)

                total_samples = 0
                for npy_file in tf_dir.glob("X*.npy"):
                    try:
                        arr = np.load(npy_file, mmap_mode='r')
                        total_samples += arr.shape[0]
                    except Exception as exc:
                        print(f"[DEBUG] Could not load npy file {npy_file}: {exc}")

                if total_samples > 0:
                    entry = {
                        "year": year,
                        "pair": pair,
                        "timeframe": tf,
                        "count": total_samples
                    }
                    result["details"].append(entry)
                    result["samples_per_year"][year] = result["samples_per_year"].get(year, 0) + total_samples
                    result["samples_per_pair"][pair] = result["samples_per_pair"].get(pair, 0) + total_samples
                    result["samples_per_tf"][tf] = result["samples_per_tf"].get(tf, 0) + total_samples

    result["years"] = sorted(year_set)
    result["pairs"] = sorted(pair_set)
    result["timeframes"] = sorted(tf_set)
    return result


def _clean_readme(content):
    """
    Clean up the README markdown to avoid multiple consecutive blank lines.
    """
    lines = content.splitlines()
    cleaned = []
    prev_empty = False
    for line in lines:
        is_empty = (line.strip() == "")
        if is_empty and prev_empty:
            continue  # skip duplicate blank lines
        cleaned.append(line)
        prev_empty = is_empty
    # Remove trailing blank lines
    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    return "\n".join(cleaned) + "\n"


def generate_dataset_docs(output_root, final_index=None, pascal_name=None, overrides=None):
    output_root = Path(output_root)
    manifold_name = output_root.name
    if not pascal_name:
        pascal_name = manifold_name

    # Determine task
    task = None
    yaml_path = output_root / "dataset_info.yaml"
    existing_info = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing_info = yaml.safe_load(f) or {}
                cached_t = existing_info.get("task", existing_info.get("dataset_type"))
                if cached_t and cached_t != "quality":
                    task = cached_t
        except Exception as exc:
            print(f"[DEBUG] Could not read dataset_info.yaml at {yaml_path}: {exc}")

    if overrides and overrides.get('dataset_type'):
        task = overrides.get('dataset_type')
    elif overrides and overrides.get('task'):
        task = overrides.get('task')
    elif final_index and len(final_index) > 0:
        task = final_index[0].get("task", task)

    # High-precision task mapping from registered manifold specification
    if manifold_name in MANIFOLD_TASK_MAP:
        task = MANIFOLD_TASK_MAP[manifold_name]
    elif not task:
        name_lower = manifold_name.lower()
        if "forex" in name_lower:
            task = "forex"
        elif "authenticity" in name_lower:
            task = "authenticity"
        elif "restoration" in name_lower or any(x in name_lower for x in ["ffanet", "mirnet", "mprnet", "nafnet", "film", "codeformer"]):
            task = "restoration"
        elif "ultrazoom" in name_lower or "superresolution" in name_lower:
            task = "super-resolution"
        elif "parsenet" in name_lower or "segmentation" in name_lower:
            task = "segmentation"
        elif "retinaface" in name_lower or "yolo" in name_lower or "detection" in name_lower:
            task = "detection"
        elif "upn" in name_lower or "parameter" in name_lower:
            task = "parameter_prediction"
        elif "classification" in name_lower or "nsfw" in name_lower:
            task = "classification"
        else:
            task = "quality"

    task_key = str(task)   # ensure string

    # 1. index.json
    if final_index:
        with open(output_root / "index.json", "w", encoding="utf-8") as f:
            json.dump(final_index, f, indent=2)

    # 2. Count samples & sources
    sources = {}
    total_samples = 0

    if final_index and len(final_index) > 0:
        total_samples = len(final_index)
        for item in final_index:
            actual_src = item.get("source", "")
            if not actual_src or actual_src.lower() in ["old", "unknown", "none", "legacy"]:
                name_parts = item.get("name", "").split("_")
                if len(name_parts) >= 3:
                    actual_src = "_".join(name_parts[1:-1])
                else:
                    actual_src = actual_src if actual_src else "Unknown"
            src = format_source(actual_src)
            if src not in sources:
                sources[src] = {"train": 0, "val": 0, "total": 0}
            sources[src]["total"] += 1
            split = item.get("split", "unknown")
            if split in ["train", "val"]:
                sources[src][split] += 1
    elif task_key == "forex":
        forex_data = scan_forex_manifold(output_root)
        total_samples = sum(forex_data.get("samples_per_year", {}).values())
        sources = {
            "MetaTrader 5 Native Cache": {
                "train": "N/A",
                "val": "N/A",
                "total": total_samples
            }
        }
        existing_info["forex_scan"] = forex_data
        if not existing_info.get("pairs"):
            existing_info["pairs"] = list(forex_data.get("pairs", []))
        if not existing_info.get("timeframe_rungs"):
            existing_info["timeframe_rungs"] = list(forex_data.get("timeframes", []))
        if not existing_info.get("start_date"):
            existing_info["start_date"] = "2019-01-01"
        if not existing_info.get("lookback_bars"):
            existing_info["lookback_bars"] = 168
        if not existing_info.get("category"):
            existing_info["category"] = "Forex & Financial Time-Series"
        if overrides:
            existing_info.update(overrides)
    elif manifold_name in MANIFEST_CACHE:
        cache_entry = MANIFEST_CACHE[manifold_name]
        total_samples = cache_entry.get("total_samples", 0)
        cached_sources = cache_entry.get("sources", {})
        for raw_src, c_info in cached_sources.items():
            fmt = format_source(raw_src)
            sources[fmt] = {
                "train": c_info.get("train", 0),
                "val": c_info.get("val", 0),
                "total": c_info.get("total", 0)
            }
        if "task" in cache_entry:
            task_key = str(cache_entry["task"])
    elif (output_root / "index.json").exists():
        try:
            with open(output_root / "index.json", "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            total_samples = len(idx_data)
            for item in idx_data:
                actual_src = item.get("source", "")
                src = format_source(actual_src)
                if src not in sources:
                    sources[src] = {"train": 0, "val": 0, "total": 0}
                sources[src]["total"] += 1
                split = item.get("split", "unknown")
                if split in ["train", "val"]:
                    sources[src][split] += 1
        except Exception as exc:
            print(f"[DEBUG] Error tallying index sources: {exc}")
    elif existing_info and "count" in existing_info:
        total_samples = existing_info.get("count", 0)
        orig_sources = existing_info.get("original_sources", [])
        if orig_sources and total_samples > 0:
            per_src = total_samples // len(orig_sources)
            val_pct = 0.12
            for s in orig_sources:
                fmt = format_source(s)
                val_c = int(per_src * val_pct)
                train_c = per_src - val_c
                sources[fmt] = {"train": train_c, "val": val_c, "total": per_src}

    # 3. dataset_info.yaml — task-aware, preserves existing task-specific fields
    # (Phase 0 Fix C: forex metadata must survive doc regeneration)
    src_keys = list(sources.keys()) if sources else [f"{pascal_name}-source"]
    relative_path = str(Path('..') / 'LemGendaryDatasets' / output_root.name)

    info_fields: dict = {
        "count": total_samples if isinstance(total_samples, int) else 0,
        "task": task_key,
        "original_sources": src_keys,
        "path": relative_path,
        "source": f"{pascal_name}-manifold",
        "last_processed": datetime.now().isoformat(),
    }

    if task_key == "forex":
        # Preserve all task-specific forex fields from existing_info or overrides
        forex_scan_local = existing_info.get("forex_scan", {})
        if not isinstance(forex_scan_local, dict):
            forex_scan_local = {}

        pairs = (
            existing_info.get("pairs")
            or (overrides.get("pairs") if overrides else None)
            or forex_scan_local.get("pairs", [])
        )
        tfs = (
            existing_info.get("timeframe_rungs")
            or (overrides.get("timeframe_rungs") if overrides else None)
            or forex_scan_local.get("timeframes", [])
        )
        start_date = (
            existing_info.get("start_date")
            or (overrides.get("start_date") if overrides else None)
            or "2019-01-01"
        )
        lookback_bars = (
            existing_info.get("lookback_bars")
            or (overrides.get("lookback_bars") if overrides else None)
            or 168
        )
        category = (
            existing_info.get("category")
            or (overrides.get("category") if overrides else None)
            or "Forex & Financial Time-Series"
        )

        info_fields["name"] = pascal_name
        info_fields["dataset_type"] = "forex"
        info_fields["category"] = category
        info_fields["pairs"] = [str(p) for p in pairs] if pairs else []
        info_fields["timeframe_rungs"] = [int(tf) for tf in tfs] if tfs else []
        info_fields["start_date"] = str(start_date)
        info_fields["lookback_bars"] = int(lookback_bars)
        info_fields["format"] = "parquet"
        info_fields["compression"] = "zstd"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            info_fields,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # 4. category.txt
    cat_str = "General Dataset"
    if task_key in TASK_META:
        cat_str = TASK_META[task_key].get("category", "General Dataset")
    elif "detection" in TASK_META:
        cat_str = TASK_META["detection"].get("category", "General Dataset")
    if overrides and overrides.get('category'):
        cat_str = overrides['category']
    with open(output_root / "category.txt", "w", encoding="utf-8") as f:
        f.write(f"{cat_str}\n")

    # 5. classes.txt
    with open(output_root / "classes.txt", "w", encoding="utf-8") as f:
        if task_key == "forex":
            f.write("SELL\nHOLD\nBUY\n")
        else:
            class_name = "face" if task_key == "pose" else task_key
            f.write(f"{class_name}\n")

    # 6. README
    # Get task metadata safely, guarantee `m` is a dict
    m = TASK_META.get(task_key, {})
    if not isinstance(m, dict):
        m = {}

    resolved_desc = m.get('desc', 'Dataset manifold.')
    resolved_obj = m.get('obj', 'Dataset objective.')
    img_desc = "RGB"
    tgt_desc = ""

    if task_key == "restoration":
        name_lower = manifold_name.lower()
        if "dehazing" in name_lower or "indoor" in name_lower or "outdoor" in name_lower:
            task_noun = "dehazing"
            img_desc = "Hazy RGB images"
            tgt_desc = "Haze-free reference images"
            resolved_obj = "Remove haze from images and restore visual quality."
        elif "deraining" in name_lower:
            task_noun = "deraining"
            img_desc = "Rainy RGB images"
            tgt_desc = "Rain-free reference images"
            resolved_obj = "Remove rain streaks from images and restore visual quality."
        elif "deblurring" in name_lower or "debluring" in name_lower:
            task_noun = "deblurring"
            img_desc = "Blurry RGB images"
            tgt_desc = "Blur-free reference images"
            resolved_obj = "Remove blur from images and restore visual sharpness."
        elif "denoising" in name_lower:
            task_noun = "denoising"
            img_desc = "Noisy RGB images"
            tgt_desc = "Noise-free reference images"
            resolved_obj = "Remove noise from images and restore visual quality."
        elif "exposure" in name_lower or "lowlight" in name_lower:
            task_noun = "exposure correction and low-light enhancement"
            img_desc = "Under/over-exposed RGB images"
            tgt_desc = "Properly exposed reference images"
            resolved_obj = "Correct under/over-exposed images and enhance visual quality."
        elif "film" in name_lower:
            task_noun = "old film restoration"
            img_desc = "Degraded film frame RGB images"
            tgt_desc = "Restored film frame reference images"
            resolved_obj = "Restore degraded vintage film frames (scratches, noise, color fade)."
        else:
            task_noun = "restoration"
            img_desc = "Degraded RGB images"
            tgt_desc = "Clean reference images"
            resolved_obj = "Restore degraded images and enhance visual quality."
        resolved_desc = f"Standardized dataset for image {task_noun} models."

    if task_key == "forex":
        # ─── SAFETY: ensure forex_scan is a dict ──────────────────────
        forex_scan = existing_info.get("forex_scan", {})
        if not isinstance(forex_scan, dict):
            forex_scan = {}

        # Safely extract pairs and timeframes, ensuring they are lists
        pairs_raw = existing_info.get('pairs', []) or (overrides.get('pairs') if overrides else [])
        if not pairs_raw and isinstance(forex_scan, dict):
            pairs_raw = forex_scan.get('pairs', [])
        if not isinstance(pairs_raw, list):
            pairs_raw = []
        pairs_list = [str(p) for p in pairs_raw]

        tfs_raw = existing_info.get('timeframe_rungs', []) or (overrides.get('timeframe_rungs') if overrides else [])
        if not tfs_raw and isinstance(forex_scan, dict):
            tfs_raw = forex_scan.get('timeframes', [])
        if not isinstance(tfs_raw, list):
            tfs_raw = []
        tfs_list = [int(tf) for tf in tfs_raw]

        start_date_str = existing_info.get('start_date', '2019-01-01') or (overrides.get('start_date') if overrides else '2019-01-01')
        lookback_bars = existing_info.get('lookback_bars', 168) or (overrides.get('lookback_bars') if overrides else 168)
        category_str = existing_info.get('category', 'Forex & Financial Time-Series') or (overrides.get('category') if overrides else 'Forex & Financial Time-Series')

        tf_names = {1: 'M1 (1min)', 5: 'M5 (5min)', 15: 'M15 (15min)', 60: 'H1 (60min)', 240: 'H4 (240min)', 1440: 'D1 (1440min)'}
        tf_labels = [tf_names.get(tf, f'{tf}min') for tf in tfs_list]
        pairs_display = ', '.join(pairs_list) if pairs_list else 'All Primary & Secondary FX Pairs'
        tfs_display = ', '.join(tf_labels) if tf_labels else 'M1 (1min), M5 (5min), M15 (15min), H1 (60min), H4 (240min), D1 (1440min)'

        # Models
        applicable_models = []
        for m_key, m_info in MODELS_META.items():
            if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", []):
                applicable_models.append(m_info)
        if not applicable_models:
            for m_key, m_info in MODELS_META.items():
                if isinstance(m_info, dict) and m_key.lower() in manifold_name.lower().replace("lemgendized", "").replace("large", ""):
                    applicable_models.append(m_info)

        models_markdown = ""
        for am in applicable_models:
            models_markdown += f"### Model: {am.get('name', 'Unknown Model')}\n\n"
            arch_val = am.get('arch') or am.get('architecture_type') or "Standard Backbone"
            models_markdown += f"- **Architecture**: {arch_val}\n"
            models_markdown += f"- **Optimization**: {am.get('loss', 'Unknown')}\n\n"
            sota = am.get('sota_targets', {})
            if sota:
                models_markdown += "| Metric | Baseline | Advanced | SOTA |\n"
                models_markdown += "| :--- | :--- | :--- | :--- |\n"
                for met, val in sota.items():
                    met_name = met.replace('_', ' ').title().replace('Psnr', 'PSNR').replace('Ssim', 'SSIM').replace('Lpips', 'LPIPS').replace('Fid', 'FID').replace('Map', 'mAP').replace('Miou', 'mIoU')
                    if isinstance(val, (int, float)):
                        lower_is_better = any(x in met.lower() for x in ['loss', 'lpips', 'fid', 'drawdown', 'mae', 'mse', 'rank_margin'])
                        if lower_is_better:
                            base = val * 1.5
                            adv = val * 1.2
                            models_markdown += f"| **{met_name}** | < {base:.2f} | < {adv:.2f} | **< {val}** |\n"
                        elif 'psnr' in met.lower():
                            base = val * 0.85
                            adv = val * 0.94
                            models_markdown += f"| **{met_name}** | ~{base:.1f} dB | > {adv:.1f} dB | **> {val:.1f} dB** |\n"
                        elif 'ssim' in met.lower():
                            base = val * 0.88
                            adv = val * 0.95
                            models_markdown += f"| **{met_name}** | ~{base:.4f} | > {adv:.4f} | **> {val:.4f}** |\n"
                        else:
                            is_pct = any(x in met.lower() for x in ['acc', 'win', 'rate']) or (20.0 < val <= 100.0)
                            if is_pct and val > 10.0:
                                base = val * 0.8
                                adv = val * 0.9
                                models_markdown += f"| **{met_name}** | ~{base:.1f}% | > {adv:.1f}% | **> {val}%** |\n"
                            else:
                                base = val * 0.8
                                adv = val * 0.9
                                models_markdown += f"| **{met_name}** | ~{base:.2f} | > {adv:.2f} | **> {val}** |\n"
                    else:
                        models_markdown += f"| **{met_name}** | N/A | N/A | **{val}** |\n"
                models_markdown += "\n"

        if not models_markdown:
            models_markdown = "No models are explicitly bound to this dataset in models_metadata.yaml.\n"

        # Year table – now safely getting details
        details = forex_scan.get("details", []) if isinstance(forex_scan, dict) else []
        year_table = ""
        if details:
            year_rows = {}
            for d in details:
                key = (d["year"], d["pair"], d["timeframe"])
                year_rows[key] = year_rows.get(key, 0) + d["count"]
            year_table = "| Year | Pair | Timeframe | Samples |\n"
            year_table += "| :--- | :--- | :--- | :--- |\n"
            for (year, pair, tf), cnt in sorted(year_rows.items()):
                year_table += f"| {year} | {pair} | {tf}min | {cnt:,} |\n"
        else:
            year_table = "| Year | Pair | Timeframe | Samples |\n"
            year_table += "| :--- | :--- | :--- | :--- |\n"
            year_table += "| (dynamic) | (dynamic) | (dynamic) | (dynamic) |\n"

        # Structure lines
        structure_lines = []
        for item in sorted(output_root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            name = item.name
            if name.endswith("_colab_training.ipynb"):
                desc = "Auto-generated Google Colab notebook for cloud training."
            elif "_training" in name and name.endswith(".ipynb"):
                desc = "Auto-generated Jupyter notebook for model training."
            elif name.startswith("ForexUniverse"):
                if name.endswith(".parquet"):
                    desc = f"Year‑chunked unified Parquet manifold: {name}"
                else:
                    desc = f"Year‑chunked shard directory: {name}"
            elif name == "category.txt":
                desc = "Top-level categorization tag."
            elif name == "classes.txt":
                desc = "Class labels mapping."
            elif name == "dataset_info.yaml":
                desc = "Manifest metadata for automated PyTorch loaders."
            elif name == "dataset-metadata.json":
                desc = "Kaggle Frictionless metadata manifest, licensing, and schema column definitions for Parquet feature tensors."
            elif name == "README.md":
                desc = "This documentation file."
            else:
                desc = "Dataset component."
            if item.is_dir():
                structure_lines.append(f"- **`{name}/`**: {desc}")
            else:
                structure_lines.append(f"- **`{name}`**: {desc}")
        structure_text = "\n".join(structure_lines)

        readme = f"""# {manifold_name}

> High-fidelity OHLCV temporal manifold for training multi-scale financial prediction models.

## Dataset Overview

- **Category:** {category_str}
- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic Multi-Regime Generator
- **Pairs Included:** {pairs_display}
- **Timeframe Rungs:** {tfs_display}
- **Historical Horizon:** {start_date_str} to Present (6-Fold Walk-Forward Matrix with 14-day Embargo)
- **Lookback Window:** {lookback_bars} bars
- **Total Samples:** {total_samples:,}
- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip Target Heads (TP/SL)
- **Architecture Base:** Causal TCN + Cross-Timeframe Multi-Head Attention
- **Primary Task:** Predict directional probability (Sell/Hold/Buy) and regress optimal Take-Profit/Stop-Loss boundaries.

## Year‑Chunked Shard Breakdown

The dataset is organised by year into unified Apache Parquet files (`ForexUniverseYYYY.parquet`), each containing all pairs and timeframes for that year with Zstandard compression.

{year_table}

## Model Training Profiles

{models_markdown}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{manifold_name.lower().replace('_', '-')})
"""
    else:
        # Non-Forex README
        table_rows = []
        total_train_all = 0
        total_val_all = 0

        for src, counts in sorted(sources.items(), key=lambda x: (x[1].get('total', 0) if isinstance(x[1].get('total'), int) else 0), reverse=True):
            tr = counts.get('train', 0)
            vl = counts.get('val', 0)
            tot = counts.get('total', 0)
            if isinstance(tr, int):
                total_train_all += tr
            if isinstance(vl, int):
                total_val_all += vl
            tr_str = f"{tr:,}" if isinstance(tr, int) else tr
            vl_str = f"{vl:,}" if isinstance(vl, int) else vl
            tot_str = f"{tot:,}" if isinstance(tot, int) else tot
            table_rows.append(f"| **{src}** | {tr_str} | {vl_str} | {tot_str} samples |")

        table_text = "\n".join(table_rows)
        if not table_text:
            table_text = "| **Standard Synthesis** | N/A | N/A | Full Contribution |"

        total_samples_display = f"{total_samples:,}" if isinstance(total_samples, int) else str(total_samples)
        arch_base = TASK_ARCH_BASE.get(task_key, "Deep Convolutional / Transformer Architecture")

        if total_train_all == 0 and isinstance(total_samples, int) and total_samples > 0:
            total_val_all = int(total_samples * 0.12)
            total_train_all = total_samples - total_val_all

        manifest_rows = []
        manifest_rows.append(f"| **images** | {total_train_all:,} | {total_val_all:,} |")
        if (output_root / "targets").exists() or task_key in ["restoration", "super-resolution"]:
            manifest_rows.append(f"| **targets** | {total_train_all:,} | {total_val_all:,} |")
        if (output_root / "labels").exists() or task_key in ["detection", "pose", "classification"]:
            manifest_rows.append(f"| **labels** | {total_train_all:,} | {total_val_all:,} |")
        manifest_text = "\n".join(manifest_rows)

        # Safe targets description
        targets_desc = m.get('targets_desc', "Target matrices or masks for training.") if isinstance(m, dict) else "Target matrices or masks for training."

        desc_map = {
            "images": f"Normalized input tensors ({img_desc}, standardized resolution).",
            "labels": "Strict numerical annotation vectors (JSON/TXT format).",
            "targets": f"Clean ground truth tensors ({tgt_desc})." if task_key == "restoration" else targets_desc,
            "shards": "WebDataset `.tar` shards containing serialized manifold data.",
            "forex": "Shards containing serialized manifold data.",
            "dataset_info.yaml": "Manifest metadata for automated PyTorch loaders.",
            "dataset-metadata.json": "Kaggle Frictionless metadata manifest, licensing, and schema column definitions for Parquet feature tensors.",
            "category.txt": "Top-level categorization tag.",
            "classes.txt": "Class labels mapping.",
            "index.json": "Compiled metadata index mapping all dataset samples.",
            "README.md": "This documentation file."
        }

        structure_lines = []
        if output_root.exists():
            for item in sorted(output_root.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                name = item.name
                if name.endswith("_colab_training.ipynb"):
                    desc = "Auto-generated Google Colab notebook for cloud training."
                elif "_training" in name and name.endswith(".ipynb"):
                    desc = "Auto-generated Jupyter notebook for model training."
                elif name.endswith("_usage.ipynb"):
                    desc = "Auto-generated notebook demonstrating standalone model inference."
                else:
                    desc = desc_map.get(name, "Dataset component.")
                if item.is_dir():
                    structure_lines.append(f"- **`{name}/`**: {desc}")
                else:
                    structure_lines.append(f"- **`{name}`**: {desc}")

        structure_text = "\n".join(structure_lines)

        # Models
        applicable_models = []
        for m_key, m_info in MODELS_META.items():
            if isinstance(m_info, dict) and manifold_name in m_info.get("datasets", []):
                applicable_models.append(m_info)
        if not applicable_models:
            for m_key, m_info in MODELS_META.items():
                if isinstance(m_info, dict) and m_key.lower() in manifold_name.lower().replace("lemgendized", "").replace("large", ""):
                    applicable_models.append(m_info)

        models_markdown = ""
        for am in applicable_models:
            models_markdown += f"### Model: {am.get('name', 'Unknown Model')}\n\n"
            arch_val = am.get('arch') or am.get('architecture_type') or "Standard Backbone"
            models_markdown += f"- **Architecture**: {arch_val}\n"
            models_markdown += f"- **Optimization**: {am.get('loss', 'Unknown')}\n\n"
            sota = am.get('sota_targets', {})
            if sota:
                models_markdown += "| Metric | Baseline | Advanced | SOTA |\n"
                models_markdown += "| :--- | :--- | :--- | :--- |\n"
                for met, val in sota.items():
                    met_name = met.replace('_', ' ').title().replace('Psnr', 'PSNR').replace('Ssim', 'SSIM').replace('Lpips', 'LPIPS').replace('Fid', 'FID').replace('Map', 'mAP').replace('Miou', 'mIoU')
                    if isinstance(val, (int, float)):
                        lower_is_better = any(x in met.lower() for x in ['loss', 'lpips', 'fid', 'drawdown', 'mae', 'mse', 'rank_margin'])
                        if lower_is_better:
                            base = val * 1.5
                            adv = val * 1.2
                            models_markdown += f"| **{met_name}** | < {base:.2f} | < {adv:.2f} | **< {val}** |\n"
                        elif 'psnr' in met.lower():
                            base = val * 0.85
                            adv = val * 0.94
                            models_markdown += f"| **{met_name}** | ~{base:.1f} dB | > {adv:.1f} dB | **> {val:.1f} dB** |\n"
                        elif 'ssim' in met.lower():
                            base = val * 0.88
                            adv = val * 0.95
                            models_markdown += f"| **{met_name}** | ~{base:.4f} | > {adv:.4f} | **> {val:.4f}** |\n"
                        else:
                            is_pct = any(x in met.lower() for x in ['acc', 'win', 'rate']) or (20.0 < val <= 100.0)
                            if is_pct and val > 10.0:
                                base = val * 0.8
                                adv = val * 0.9
                                models_markdown += f"| **{met_name}** | ~{base:.1f}% | > {adv:.1f}% | **> {val}%** |\n"
                            else:
                                base = val * 0.8
                                adv = val * 0.9
                                models_markdown += f"| **{met_name}** | ~{base:.2f} | > {adv:.2f} | **> {val}** |\n"
                    else:
                        models_markdown += f"| **{met_name}** | N/A | N/A | **{val}** |\n"
                models_markdown += "\n"

        if not models_markdown:
            models_markdown = "No models are explicitly bound to this dataset in models_metadata.yaml.\n"

        # Safe category display
        category = m.get('category', 'Dataset') if isinstance(m, dict) else 'Dataset'

        readme = f"""# {manifold_name}

> {resolved_desc}

## Dataset Overview

- **Category:** {category}
- **Total Samples:** {total_samples_display}
- **Architecture Base:** {arch_base}
- **Primary Task:** {resolved_obj}

## Composition & Lineage

This manifold is a high-fidelity merge of the following original sources:

| Source Dataset | Train | Val | Total Contribution |
| :--- | :--- | :--- | :--- |
{table_text}

## Model Training Profiles

{models_markdown}

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

{structure_text}

## Physical Data Manifest

| Folder | Train | Val |
| :--- | :--- | :--- |
{manifest_text}

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/{manifold_name.lower().replace('_', '-')})
"""

    # ─── Clean up blank lines ──────────────────────────────────────────
    readme = _clean_readme(readme)

    with open(output_root / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # ─── Generate dataset-metadata.json for Kaggle ──────────────────────
    slug = manifold_name.lower().replace("_", "")
    resources = []
    is_forex = (task_key == "forex")
    if is_forex:
        for y in range(2019, 2027):
            resources.append({
                "path": f"{manifold_name}/ForexUniverse{y}.parquet",
                "description": f"Annual OHLCV and feature tensor shards for year {y}",
                "schema": {
                    "fields": FOREX_COLUMN_FIELDS
                }
            })
            resources.append({
                "path": f"ForexUniverse{y}.parquet",
                "description": f"Annual OHLCV and feature tensor shards for year {y}",
                "schema": {
                    "fields": FOREX_COLUMN_FIELDS
                }
            })

    subtitle = f"High-fidelity manifold for {cat_str} machine learning models"
    if len(subtitle) > 80:
        subtitle = f"High-fidelity manifold for {cat_str} models"
    if len(subtitle) > 80:
        subtitle = subtitle[:77] + "..."
    if len(subtitle) < 20:
        subtitle = "High-fidelity machine learning training manifold"

    metadata_payload = {
        "title": manifold_name.replace("Large", "").replace("LemGendized", "LemGendized "),
        "id": f"lemtreursi/{slug}",
        "subtitle": subtitle,
        "description": readme,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": resources
    }
    with open(output_root / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return total_samples


def regenerate_all_docs(datasets_dir=None):
    if datasets_dir is None:
        datasets_dir = Path(__file__).parent.parent / "LemGendaryDatasets"
    datasets_dir = Path(datasets_dir)

    print(f"Scanning manifolds in {datasets_dir}...")

    # Discover all target manifolds from unified_data.yaml
    prefix = UNIFIED_DATA.get("_registry_metadata", {}).get("name_prefix", "LemGendized")
    suffix = UNIFIED_DATA.get("_registry_metadata", {}).get("name_suffix", "Large")
    target_names = set()
    for d_key, d_info in UNIFIED_DATA.get("datasets", {}).items():
        t_name = d_info.get("name", d_key)
        target_names.add(f"{prefix}{t_name}{suffix}")

    # Also include any existing folders in LemGendaryDatasets
    if datasets_dir.exists():
        for p in datasets_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                target_names.add(p.name)

    count = 0
    for name in sorted(target_names):
        p = datasets_dir / name
        p.mkdir(parents=True, exist_ok=True)
        print(f"Regenerating docs for {name}...")
        try:
            samples = generate_dataset_docs(p, None, name)
            print(f"  Success: {name} -> Total Samples: {samples}")
            count += 1
        except Exception as e:
            print(f"  Error on {name}: {e}")
    print(f"Regeneration complete for {count} manifolds.")


regenerate_all_non_forex = regenerate_all_docs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LemGendary Dataset Doc Generator")
    parser.add_argument("--all", action="store_true", help="Regenerate all manifold READMEs")
    parser.add_argument("--manifold", type=str, default=None, help="Specific manifold folder name")
    args = parser.parse_args()

    if args.manifold:
        m_path = Path(__file__).parent.parent / "LemGendaryDatasets" / args.manifold
        generate_dataset_docs(m_path, None, args.manifold)
    else:
        regenerate_all_docs()