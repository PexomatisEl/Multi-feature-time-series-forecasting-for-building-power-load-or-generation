# ⚡ Microgrid Load Forecasting: A Multi-Horizon Foundation Model Approach

This repository contains the complete codebase and experimental pipeline for a Master's Thesis evaluating classical machine learning, deep learning, and zero-shot foundation models (LLMs) in the context of electrical microgrid load forecasting.

## 📖 Project Overview

Accurate load forecasting is critical for the stability and economic optimization of decentralized microgrids. This project explores how temporal data aggregation (smoothing micro-volatility) affects algorithmic performance across three distinct operational horizons: **15-minute, 30-minute, and 1-hour intervals**. 

The study benchmarks classical tree-based ensembles and advanced neural networks against the latest generative time series foundation models (Amazon Chronos T5).

### 🚀 Key Engineering Highlights
* **Physics-Based Scaling:** Target variables were mathematically converted from cumulative Energy (kWh) to Average Power (kW) prior to temporal aggregation, guaranteeing mathematical scale parity across all horizons.
* **Causal Imputation Cascade:** Developed a custom 3-Tier algorithm to patch IoT sensor outages (Linear Interpolation for micro-gaps, Homologous Shift for meso-gaps, Macro-Patching for structural outages).
* **Direct Multi-Step Forecasting (MIMO):** Avoided recursive error accumulation by wrapping classical models in a Multi-Output regressor, forcing all models to output a synchronized 24-hour day-ahead trajectory.
* **Chronos Foundation Model Scenario Testing:** Evaluated the *Chronos-2 Multivariate* LLM under three rigorous stress tests: **Oracle** (perfect foresight), **Realistic** (Gaussian noise injected into weather/market predictions), and **Blind** (purely autoregressive).
* **Statistical Rigor:** Implemented the Diebold-Mariano test with Benjamini-Hochberg False Discovery Rate (FDR) correction to definitively prove mathematical superiority.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                 # Raw IoT telemetry (Load, PV, Weather, SMP)
│   └── processed/           # Imputed, scaled, and feature-engineered parquets
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb       # Grid alignment, anomaly neutralization
│   ├── 02_Model_Training_15m.ipynb     # 96-step sequence forecasting
│   ├── 03_Model_Training_30m.ipynb     # 48-step sequence forecasting
│   └── 04_Model_Training_1h.ipynb      # 24-step sequence forecasting
├── src/
│   ├── config.py                # Centralized hyperparameters & seeds
│   ├── evaluation_utils.py      # MASE calculations, Diebold-Mariano, Residual Diagnostics
│   ├── ml_baselines.py          # Optuna-optimized XGBoost & Random Forest
│   ├── model_architectures.py   # PatchTST, Vanilla Transformer, LSTM, Chronos Pipelines
│   └── model_prep_tools.py      # T-Zero Slicing, 3D Tensor generation, Feature Eng
└── results/                 # Cached serialized models (.joblib, .keras), inferences (.npz)