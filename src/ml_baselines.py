import time
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import src.config as config
from src.evaluation_utils import calculate_metrics, plot_predictions
from sklearn.multioutput import MultiOutputRegressor
import gc
import os
import joblib



def xgboost_implementation(X_train, y_train, X_val, y_val, X_test, y_test, 
                      horizon_name, y_train_full=None, target_scaler=None,
                      target_feature_idx=0, optimization_metric="MAE",
                      train_from_scratch=True):
    
    horizon = config.HORIZON_STEPS[horizon_name]
    model_name = f"XGBoost_{optimization_metric}"
    print(f"\nRUNNING {model_name.upper()} ({horizon_name}) | Horizon: {horizon}")
    
    start_time = time.time()
    
    if len(X_train.shape) == 3:
        print("Performing Smart Flattening (Feature Reduction)")
        def smart_flatten(X_3d):
            hist_target = X_3d[:, :, target_feature_idx] 
            covariate_indices = [i for i in range(X_3d.shape[2]) if i != target_feature_idx]
            current_state = X_3d[:, -1, covariate_indices] 
            return np.hstack((hist_target, current_state))
            
        X_train = smart_flatten(X_train)
        X_val = smart_flatten(X_val)
        X_test = smart_flatten(X_test)
        print(f"Reduced feature columns shape to {X_train.shape[1]}")
    
    model_save_path = f"../results/{horizon_name}/{model_name}/{model_name.lower()}_model.joblib"
    best_params = {}
    final_model = None
    
    if not train_from_scratch and os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}")
        final_model = joblib.load(model_save_path)
    else:
        subset_size = int(len(X_train) * 0.50)
        np.random.seed(config.RANDOM_SEED)
        idx_train = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_sub = X_train[idx_train]
        y_train_sub = y_train[idx_train]

        val_subset_size = int(len(X_val) * 0.50)
        idx_val = np.random.choice(len(X_val), val_subset_size, replace=False)
        X_val_sub = X_val[idx_val]
        y_val_sub = y_val[idx_val]

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
            
            if optimization_metric == "MAE":
                param['objective'] = 'reg:absoluteerror'
                base_regressor = xgboost.XGBRegressor(**param)
                model = MultiOutputRegressor(base_regressor)
                model.fit(X_train_sub, y_train_sub)
                preds = model.predict(X_val_sub)
                return mean_absolute_error(y_val_sub, preds)
            else:
                base_regressor = xgboost.XGBRegressor(**param)
                base_regressor.fit(X_train_sub, y_train_sub)
                preds = base_regressor.predict(X_val_sub)
                return np.sqrt(mean_squared_error(y_val_sub, preds))
        
        print(f"Identifying best XGBoost hyperparameters using Optuna (Optimizing for {optimization_metric})")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
        study.optimize(objective, n_trials=config.XGBOOST_OPTUNA_TRIALS, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cuda'
        best_params['random_state'] = config.RANDOM_SEED
        best_params['n_jobs'] = -1
        if optimization_metric == "MAE":
            best_params['objective'] = 'reg:absoluteerror'
        
        print("Training Final XGBoost Model on Full Combined Dataset")
        X_train_full_combined = np.vstack((X_train, X_val))
        y_train_full_combined = np.vstack((y_train, y_val))
        
        if optimization_metric == "MAE":
            final_model = MultiOutputRegressor(xgboost.XGBRegressor(**best_params))
        else:
            final_model = xgboost.XGBRegressor(**best_params)
            
        final_model.fit(X_train_full_combined, y_train_full_combined)
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Model successfully cached at {model_save_path}")
    
    test_preds = final_model.predict(X_test)
    execution_time = time.time() - start_time
    
    y_train_for_mase = y_train_full if y_train_full is not None else y_train
    
    if target_scaler is not None:
        orig_shape = test_preds.shape
        test_preds = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(orig_shape)
        y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(orig_shape)
        y_train_flat = target_scaler.inverse_transform(np.array(y_train_for_mase).reshape(-1, 1)).flatten()
    else:
        y_test_unscaled = y_test
        y_train_flat = np.array(y_train_for_mase).flatten()
        
    metrics_dict = calculate_metrics(y_test_unscaled, test_preds, model_name, execution_time, 
                                     horizon=horizon, y_train=y_train_flat)
    
    plot_predictions(y_test_unscaled, test_preds, model_name, horizon)
    
    return {
        "model": final_model,
        "predictions": test_preds,
        "y_test_real": y_test_unscaled,
        "metrics": metrics_dict,
        "best_params": best_params
    }

def rf_implementation(X_train, y_train, X_val, y_val, X_test, y_test, 
                            horizon_name, y_train_full=None, target_scaler=None, 
                            target_feature_idx=0, train_from_scratch=True):
    
    horizon = config.HORIZON_STEPS[horizon_name]
    model_name = "Random_Forest"
    print(f"\n RUNNING {model_name.upper()} ({horizon_name}) | Horizon: {horizon}")
    
    start_time = time.time()
    
    # SMART FLATTENING
    if len(X_train.shape) == 3:
        print(" -> Performing Smart Flattening (Domain Knowledge Feature Reduction)")
        def smart_flatten(X_3d):
            hist_target = X_3d[:, :, target_feature_idx] 
            covariate_indices = [i for i in range(X_3d.shape[2]) if i != target_feature_idx]
            current_state = X_3d[:, -1, covariate_indices] 
            return np.hstack((hist_target, current_state))
            
        X_train = smart_flatten(X_train)
        X_val = smart_flatten(X_val)
        X_test = smart_flatten(X_test)
        print(f"    Reduced feature columns shape to {X_train.shape[1]}")

    model_save_path = f"../results/{horizon_name}/{model_name}/{model_name.lower()}_model.joblib"
    best_params = {}
    final_model = None
    
    # So we dont have to retrain
    if not train_from_scratch and os.path.exists(model_save_path):
        print(f"Loading pre-trained model from {model_save_path}")
        final_model = joblib.load(model_save_path)
    else:
        # Sub-sample data to 10% for Optuna speed
        subset_size = int(len(X_train) * 0.50)
        np.random.seed(config.RANDOM_SEED)
        idx_train = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_sub = X_train[idx_train]
        y_train_sub = y_train[idx_train]

        val_subset_size = int(len(X_val) * 0.50)
        idx_val = np.random.choice(len(X_val), val_subset_size, replace=False)
        X_val_sub = X_val[idx_val]
        y_val_sub = y_val[idx_val]

        # OPTUNA SEARCH
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', *config.RANDOM_FOREST_SEARCH_SPACE['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *config.RANDOM_FOREST_SEARCH_SPACE['max_depth']),
                'min_samples_split': trial.suggest_int('min_samples_split', *config.RANDOM_FOREST_SEARCH_SPACE['min_samples_split']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *config.RANDOM_FOREST_SEARCH_SPACE['min_samples_leaf']),
                'max_features': trial.suggest_float('max_features', *config.RANDOM_FOREST_SEARCH_SPACE['max_features']),
                'random_state': config.RANDOM_SEED,
                'n_jobs': -1 # Use all CPU cores
            }
            
            model = RandomForestRegressor(**param)
            model.fit(X_train_sub, y_train_sub)
            
            preds = model.predict(X_val_sub)
            # Evaluating using MAE to align with the MASE leaderboard metric
            return mean_absolute_error(y_val_sub, preds)

        print("Identifying best Random Forest hyperparameters using Optuna")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
        study.optimize(objective, n_trials=config.RANDOM_FOREST_OPTUNA_TRIALS, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = config.RANDOM_SEED
        best_params['n_jobs'] = -1
        
        # FINAL TRAINING
        print("Training Final Multi-Output Random Forest on Full Dataset")
        X_train_full_combined = np.vstack((X_train, X_val))
        y_train_full_combined = np.vstack((y_train, y_val))
        
        final_model = RandomForestRegressor(**best_params)
        final_model.fit(X_train_full_combined, y_train_full_combined)
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Model successfully cached at {model_save_path}")

    # PREDICTION & INVERSE SCALING
    test_preds = final_model.predict(X_test)
    execution_time = time.time() - start_time
    
    y_train_for_mase = y_train_full if y_train_full is not None else y_train
    
    if target_scaler is not None:
        orig_shape = test_preds.shape 
        test_preds = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(orig_shape)
        y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(orig_shape)
        y_train_flat = target_scaler.inverse_transform(np.array(y_train_for_mase).reshape(-1, 1)).flatten()
    else:
        y_test_unscaled = y_test
        y_train_flat = np.array(y_train_for_mase).flatten()
    
    # EVALUATION
    metrics_dict = calculate_metrics(y_test_unscaled, test_preds, model_name, execution_time, 
                                     horizon=horizon, y_train=y_train_flat)
    
    plot_predictions(y_test_unscaled, test_preds, model_name, horizon)
    
    return {
        "model": final_model,
        "predictions": test_preds,
        "y_test_real": y_test_unscaled,
        "metrics": metrics_dict,
        "best_params": best_params
    }


def run_daily_naive(y_actual_array, horizon=96):
    """
    Day-Ahead Naive Forecast: Predicts that tomorrow's sequence 
    will look exactly like yesterday's sequence.
    """
    naive_preds = []
    
    for i in range(len(y_actual_array)):
        if i < horizon:
            # First day fallback: use the mean
            naive_preds.append(np.full(horizon, y_actual_array.mean()))
        else:
            # Predict the exact 96-step sequence from 24 hours ago
            naive_preds.append(y_actual_array[i - horizon])
            
    return np.array(naive_preds)