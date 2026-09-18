# LemGendizedForexUniverseLarge

> High-fidelity OHLCV temporal manifold for training multi-scale financial prediction models.

## Dataset Overview

- **Category:** Forex & Financial Time-Series
- **Acquisition Mode:** MetaTrader 5 Terminal API / Synthetic Multi-Regime Generator
- **Pairs Included:** All Primary & Secondary FX Pairs
- **Timeframe Rungs:** M1 (1min), M5 (5min), M15 (15min), H1 (60min), H4 (240min), D1 (1440min)
- **Historical Horizon:** 2019-01-01 to Present (6-Fold Walk-Forward Matrix with 14-day Embargo)
- **Lookback Window:** 168 bars
- **Total Samples:** 0
- **Output Classes:** `SELL` (0), `HOLD` (1), `BUY` (2) + Dual Pip Target Heads (TP/SL)
- **Architecture Base:** Causal TCN + Cross-Timeframe Multi-Head Attention
- **Primary Task:** Predict directional probability (Sell/Hold/Buy) and regress optimal Take-Profit/Stop-Loss boundaries.

## Year‑Chunked Shard Breakdown

The dataset is organised by year into unified Apache Parquet files (`ForexUniverseYYYY.parquet`), each containing all pairs and timeframes for that year with Zstandard compression.

| Year | Pair | Timeframe | Samples |
| :--- | :--- | :--- | :--- |
| (dynamic) | (dynamic) | (dynamic) | (dynamic) |

## Model Training Profiles

### Model: LemGendary Forex Predictor (Multi-Scale CNN-Transformer)

- **Architecture**: Multi-Scale CNN-Transformer (Causal TCN + Cross-Timeframe Attention)
- **Optimization**: forex_dual

| Metric | Baseline | Advanced | SOTA |
| :--- | :--- | :--- | :--- |
| **Dir Acc** | ~46.8% | > 52.6% | **> 58.5%** |
| **Win Rate** | ~44.8% | > 50.4% | **> 56.0%** |
| **Profit Factor** | ~1.32 | > 1.48 | **> 1.65** |
| **Sharpe Ratio** | ~1.48 | > 1.67 | **> 1.85** |
| **Sortino Ratio** | ~1.68 | > 1.89 | **> 2.1** |
| **Max Drawdown** | < 18.00 | < 14.40 | **< 12.0** |
| **Tp Mae** | < 18.75 | < 15.00 | **< 12.5** |
| **Sl Mae** | < 18.75 | < 15.00 | **< 12.5** |

## Repository Structure

Standardized directory logic for seamless integration into the **LemGendary Training Suite**.

- **`category.txt`**: Top-level categorization tag.
- **`classes.txt`**: Class labels mapping.
- **`dataset-metadata.json`**: Kaggle Frictionless metadata manifest, licensing, and schema column definitions for Parquet feature tensors.
- **`dataset_info.yaml`**: Manifest metadata for automated PyTorch loaders.
- **`README.md`**: This documentation file.

---

**Kaggle Native Source**: [Access Dataset](https://www.kaggle.com/datasets/lemtreursi/lemgendizedforexuniverselarge)
