"""
Centralized Configuration for Thesis Experiments
Ensures reproducibility and consistency across all models
"""

import numpy as np
import random
import tensorflow as tf
import torch

# TARGET VARIABLE
TARGET_COL = 'energy_consumption'

# RANDOM SEED & REPRODUCIBILITY
RANDOM_SEED = 2026

def set_global_seeds(seed=RANDOM_SEED):
    """
    Set all random seeds for complete reproducibility.
    Call this at the start of your notebooks.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seeds set to {seed}")


# DATA SPLITS (UNIFORM ACROSS ALL MODELS)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.10
TEST_SPLIT = 0.20

assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0, "Splits must sum to 1.0"


# HORIZONS & FREQUENCIES
ACTIVE_HORIZON = "15_min" # Options: "15_min", "30_min", "1_hour"

FORECAST_HORIZONS = {
    "15_min": 15,   # 96 steps (15 min * 96 = 1440 min = 24h)
    "30_min": 30,   # 48 steps
    "1_hour": 60,   # 24 steps
}

FORECAST_HORIZON_HOURS = 24
HORIZON_STEPS = {}
for name, freq_min in FORECAST_HORIZONS.items():
    steps = int((FORECAST_HORIZON_HOURS * 60) / freq_min)
    HORIZON_STEPS[name] = steps

# Validation
assert HORIZON_STEPS["15_min"] == 96, "15-min horizon should be 96 steps"
assert HORIZON_STEPS["30_min"] == 48, "30-min horizon should be 48 steps"
assert HORIZON_STEPS["1_hour"] == 24, "1-hour horizon should be 24 steps"

NATIVE_FREQUENCY_MINUTES = FORECAST_HORIZONS[ACTIVE_HORIZON]
SEQUENCE_LENGTH = HORIZON_STEPS[ACTIVE_HORIZON]

# Validation: Sequence length should be exactly 1 full day
SEQUENCE_HOURS = (SEQUENCE_LENGTH * NATIVE_FREQUENCY_MINUTES) / 60
assert SEQUENCE_HOURS == 24, f"Sequence length is broken: {SEQUENCE_HOURS}h != 24h"


# FEATURE ENGINEERING (MODEL-SPECIFIC)

CALENDAR_COLS = [
    'hour_sin', 'hour_cos', 
    'day_sin', 'day_cos', 
    'month_sin', 'month_cos', 
    'is_holiday', 'is_weekend', 'is_non_working_day'
]

FEATURE_SETS = {
    "TREE_MODELS": CALENDAR_COLS + [
        'temp_x_hour',
        'temperature_v1', 
        'relative_humidity_v1',
        'wind_speed_v1',
        'clouds_v1',
        'energy_consumption_lag_1w',
        'temperature_v1_lag_1w',
        'smp_lag_1w'
    ],
    
    "DEEP_LEARNING": CALENDAR_COLS + [
        'temp_x_hour',
        'temperature_v1', 
        'relative_humidity_v1',
        'wind_speed_v1',
        'clouds_v1',
        'smp',
        'energy_generation',
        'energy_flow_quarter' 
    ],
    
    "CHRONOS_COVARIATES": CALENDAR_COLS + [
        'energy_consumption_lag_1w', 
        'temperature_v1_lag_1w',
        'smp_lag_1w',
        'energy_generation_lag_1w'
    ]
}


# NEURAL NETWORK HYPERPARAMETERS
LSTM_CONFIG = {
    "hidden_units": [64, 32],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "loss": "huber",  
    "huber_delta": 0.05,
}

TRANSFORMER_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "ffn_dim": 128,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "loss": "huber",
    "huber_delta": 0.05,
}

PATCHTST_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "ffn_dim": 128,
    "patch_len": 8,
    "stride": 8,
    "dropout": 0.3,
    "learning_rate": 5e-4,
    "loss": "huber",
    "huber_delta": 0.05,
}

CHRONOS_CONFIG = {
    "model_name": "amazon/chronos-t5-large",
    "batch_size": 4,
}

CHRONOS_2_CONFIG = {
    "model_name": "autogluon/chronos-2-small",
    "batch_size": 8,
    "context_length": 512,
    "quantiles": [0.1, 0.5, 0.9]
}

TRAINING_PARAMS = {
    "epochs": 50,
    "batch_size": 32,
    "early_stopping_patience": 12,
    "lr_reduce_patience": 10,
    "lr_reduce_factor": 0.7,
    "min_learning_rate": 1e-5,
    "validation_split": 0.125,
}


# TREE-BASED MODEL HYPERPARAMETERS (Optuna Search Space)
XGBOOST_SEARCH_SPACE = {
    "max_depth": (4, 8),
    "learning_rate": (0.05, 0.2),
    "n_estimators": (100, 200),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.7, 1.0),
}

XGBOOST_OPTUNA_TRIALS = 20

RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": (100, 300),
    "max_depth": (5, 25),
    "min_samples_split": (2, 15),
    "min_samples_leaf": (1, 10),
    "max_features": (0.3,1.0)
}

RANDOM_FOREST_OPTUNA_TRIALS = 20


# SCALING STRATEGY
SCALING_METHOD = "standard" 


# METRICS EPSILON
METRICS_EPSILON_SCALE = 1e-8


# CAUSAL IMPUTATION PARAMETERS
IMPUTATION_BINS = {
    "bin1_threshold_hours": 2,      
    "bin2_threshold_hours": 168,    
    "macro_patch_weeks": 52,        
    "macro_patch_weeks_alt": 4,     
}


# VALIDATION & ASSERTIONS
def validate_config():
    """Run all configuration validations."""
    assert 0 < TRAIN_SPLIT < 1, "Train split must be in (0, 1)"
    assert 0 < VAL_SPLIT < 1, "Val split must be in (0, 1)"
    assert 0 < TEST_SPLIT < 1, "Test split must be in (0, 1)"
    assert SEQUENCE_LENGTH > 0, "Sequence length must be positive"
    for name, steps in HORIZON_STEPS.items():
        assert steps > 0, f"Horizon {name} must have positive steps"
    assert LSTM_CONFIG["dropout"] >= 0 and LSTM_CONFIG["dropout"] < 1
    assert TRANSFORMER_CONFIG["dropout"] >= 0 and TRANSFORMER_CONFIG["dropout"] < 1
    assert TRAINING_PARAMS["early_stopping_patience"] > 0
    print("Configuration validation passed")


if __name__ == "__main__":
    validate_config()
    print("\n Config loaded successfully")
    print(f" Horizons: {list(HORIZON_STEPS.keys())}")
    print(f"Train/Val/Test: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"Random seed: {RANDOM_SEED}")