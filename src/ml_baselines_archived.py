import time
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
import src.config as config
from src.evaluation_utils import calculate_metrics, plot_predictions
from sklearn.multioutput import MultiOutputRegressor
import gc



def xgboost_implementation(X_train, y_train, X_val, y_val, X_test, y_test, 
                         horizon_name, y_train_full=None, target_scaler=None):
    
    horizon = config.HORIZON_STEPS[horizon_name]
    print(f"\n{'='*50}\n RUNNING XGBOOST ({horizon_name}) | Horizon: {horizon}\n{'='*50}")
    
    start_time = time.time()
    
    # Flatten 3D to 2D
    if len(X_train.shape) == 3:
        print(" -> Flattening 3D sequence inputs to 2D")
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Sub-sample data to 10% for Optuna speed
    subset_size = int(len(X_train) * 0.10)
    np.random.seed(config.RANDOM_SEED)
    idx_train = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_sub = X_train[idx_train]
    y_train_sub = y_train[idx_train]

    val_subset_size = int(len(X_val) * 0.10)
    idx_val = np.random.choice(len(X_val), val_subset_size, replace=False)
    X_val_sub = X_val[idx_val]
    y_val_sub = y_val[idx_val]

    # Optuna Objective Function 
    def objective(trial):
        param = {
            'tree_method': 'hist',
            'device': 'cuda', 
            'n_estimators': trial.suggest_int('n_estimators', *config.XGBOOST_SEARCH_SPACE['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *config.XGBOOST_SEARCH_SPACE['max_depth']),        
            'learning_rate': trial.suggest_float('learning_rate', *config.XGBOOST_SEARCH_SPACE['learning_rate'], log=True),
            'subsample': trial.suggest_float('subsample', *config.XGBOOST_SEARCH_SPACE['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *config.XGBOOST_SEARCH_SPACE['colsample_bytree']),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1
        }
        
        model = xgboost.XGBRegressor(**param)
        model.fit(X_train_sub, y_train_sub) 
        
        preds = model.predict(X_val_sub)
        return np.sqrt(mean_squared_error(y_val_sub, preds))
    
    # Optuna Search
    print(f"Identifying best XGBoost hyperparameters using Optuna")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=config.XGBOOST_OPTUNA_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['device'] = 'cuda'
    best_params['random_state'] = config.RANDOM_SEED
    best_params['n_jobs'] = -1
    
    # Final Training on Full Dataset
    print(f"Training Final XGBoost Model on Full Dataset...")
    X_train_full_combined = np.vstack((X_train, X_val))
    y_train_full_combined = np.vstack((y_train, y_val))
    
    final_model = xgboost.XGBRegressor(**best_params)
    final_model.fit(X_train_full_combined, y_train_full_combined)
    
    # Generate Predictions
    test_preds = final_model.predict(X_test)
    execution_time = time.time() - start_time
    

    # INVERSE SCALING SHAPE RETENTION
  
    y_train_for_mase = y_train_full if y_train_full is not None else y_train
    
    if target_scaler is not None:
        orig_shape = test_preds.shape # Must be (samples, 96)
        
        # Scale and reshape back to 2D
        test_preds = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(orig_shape)
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(orig_shape)
        
        # It's okay to flatten y_train here because calculate_metrics handles it
        y_train_flat = target_scaler.inverse_transform(np.array(y_train_for_mase).reshape(-1, 1)).flatten()
    else:
        y_train_flat = np.array(y_train_for_mase).flatten()
    
    # Calculate Metrics & Plot
    metrics_dict = calculate_metrics(y_test, test_preds, "XGBoost", execution_time, 
                                     horizon=horizon, y_train=y_train_flat)
    
   
    plot_predictions(y_test, test_preds, "XGBoost", horizon)
    
    return {
        "model": final_model,
        "predictions": test_preds,
        "y_test_real": y_test,
        "metrics": metrics_dict,
        "best_params": best_params
    }

def rf_implementation(X_train, y_train, X_val, y_val, X_test, y_test, 
                      horizon_name, y_train_full=None, target_scaler=None):

    horizon = config.HORIZON_STEPS[horizon_name]
    print(f"\n{'='*50}\n RUNNING RANDOM FOREST ({horizon_name}) | Horizon: {horizon}\n{'='*50}")
    
    start_time = time.time()
    
    # Flatten 3D to 2D
    if len(X_train.shape) == 3:
        print(" -> Flattening 3D sequence inputs to 2D...")
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Optuna Objective Function
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', *config.RANDOM_FOREST_SEARCH_SPACE['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *config.RANDOM_FOREST_SEARCH_SPACE['max_depth']),
            'min_samples_split': trial.suggest_int('min_samples_split', *config.RANDOM_FOREST_SEARCH_SPACE['min_samples_split']),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *config.RANDOM_FOREST_SEARCH_SPACE['min_samples_leaf']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**param)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    
    # Optuna Search
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=config.RANDOM_FOREST_OPTUNA_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params['random_state'] = config.RANDOM_SEED
    best_params['n_jobs'] = -1
    
    X_train_full_combined = np.vstack((X_train, X_val))
    y_train_full_combined = np.vstack((y_train, y_val))
    
    final_model = RandomForestRegressor(**best_params)
    print("Training final Multi-Output Random Forest model...")
    final_model.fit(X_train_full_combined, y_train_full_combined)
    
    test_preds = final_model.predict(X_test)
    execution_time = time.time() - start_time
    
    # Inverse Transform to Real kWh for Metrics Calculations

    y_train_for_mase = y_train_full if y_train_full is not None else y_train
    
    if target_scaler is not None:
        test_preds = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_train_flat = target_scaler.inverse_transform(np.array(y_train_for_mase).reshape(-1, 1)).flatten()
    else:
        y_train_flat = np.array(y_train_for_mase).flatten()

    metrics_dict = calculate_metrics(y_test, test_preds, "Random_Forest", execution_time, 
                                     horizon=horizon, y_train=y_train_flat)
    plot_predictions(y_test, test_preds, "Random_Forest", horizon)
    
    return {
        "model": final_model,
        "predictions": test_preds,
        "y_test_real": y_test,
        "metrics": metrics_dict,
        "best_params": best_params
    }