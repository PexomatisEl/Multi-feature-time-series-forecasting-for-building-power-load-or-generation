import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os

def stitch_2d_to_1d(array_2d, horizon):
    """
    Safely converts an overlapping 2D forecasting matrix (stride=1) 
    into a continuous 1D array by extracting non-overlapping chunks.
    Used for Plotting and CSV saving.
    """
    # If it's already 1D, just return it
    if len(array_2d.shape) == 1:
        return array_2d
        
    stitched = []
    # Jump by 'horizon' to get non-overlapping windows
    for i in range(0, len(array_2d), horizon):
        stitched.extend(array_2d[i, :])
    return np.array(stitched)

def calculate_metrics(y_true, y_pred, model_name, execution_time, horizon=96, y_train=None):
    """
    Calculates academic metrics across the entire 2D forecasting matrix.
    """
    # 1. Flatten arrays to safely calculate mathematical means/sums 
    # This prevents the len() bug and allows sklearn to handle everything natively.
    y_t = np.array(y_true).flatten()
    y_p = np.array(y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    # 2. CVRMSE (Coefficient of Variation of RMSE) - ASHRAE Standard
    mean_true = np.mean(y_t)
    cv_rmse = rmse / mean_true if mean_true != 0 else 0
    
    # 3. NMBE (Normalized Mean Bias Error) - ASHRAE Standard
    # Replaced len() with y_t.size to account for the flattened 2D matrix
    nmbe = (np.sum(y_t - y_p) / ((y_t.size - 1) * mean_true))
    
    # 4. sMAPE (Symmetric Mean Absolute Percentage Error)
    # Using np.mean instead of sum()/len() is mathematically safer and cleaner
    # Added 1e-8 to denominator to prevent division by zero
    smape = 100 * np.mean(2 * np.abs(y_p - y_t) / (np.abs(y_t) + np.abs(y_p) + 1e-8))
    
    # 5. MASE (Mean Absolute Scaled Error)
    if y_train is not None:
        y_train_arr = np.array(y_train).flatten()
        naive_errors = np.abs(y_train_arr[horizon:] - y_train_arr[:-horizon])
        denominator = np.mean(naive_errors)
    else:
        naive_errors = np.abs(y_t[horizon:] - y_t[:-horizon])
        denominator = np.mean(naive_errors)
        
    mase = mae / denominator if denominator != 0 else np.nan

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "CVRMSE": cv_rmse,
        "NMBE": nmbe,
        "sMAPE": smape,
        "MASE": mase,
        "Time (s)": execution_time
    }

def plot_predictions(y_true, y_pred, model_name, horizon, num_steps=None):
    """
    Plots the Day-Ahead trajectory. Stitches 2D arrays into continuous 1D lines.
    """
    # Convert 2D matrices to 1D continuous lines for plotting
    y_true_1d = stitch_2d_to_1d(np.array(y_true), horizon)
    y_pred_1d = stitch_2d_to_1d(np.array(y_pred), horizon)
    
    # Default to plotting 4 full horizons (e.g., 4 days) if num_steps isn't provided
    if num_steps is None:
        num_steps = horizon * 4
        
    plt.figure(figsize=(20, 5))
    plt.plot(y_true_1d[:num_steps], label='Actual Load', color='black', linewidth=1.5)
    plt.plot(y_pred_1d[:num_steps], label=f'{model_name} Prediction', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f"{model_name} Day-Ahead Performance ({horizon} steps/day)", fontsize=14)
    plt.xlabel("Time Steps (Stitched Non-Overlapping Horizons)", fontsize=12)
    plt.ylabel("Energy Consumption", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_experiment_results(results_dict, model_name, horizon_name, y_test, horizon=96):
    save_dir = f"../results/{horizon_name}/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
        
    # 1. Save the Model Weights
    if "model" in results_dict:
        model = results_dict["model"]
        if hasattr(model, 'save'):
            model_path = os.path.join(save_dir, f"{model_name.lower()}_model.keras")
            model.save(model_path)
        elif isinstance(model, str): 
            pass # Foundation models like Chronos don't need local saving
        else:
            model_path = os.path.join(save_dir, f"{model_name.lower()}_model.joblib")
            joblib.dump(model, model_path)
            
    # 2. Save Hyperparameters OR Training History
    if "best_params" in results_dict:
        params_path = os.path.join(save_dir, f"{model_name.lower()}_params.json")
        with open(params_path, "w") as f:
            json.dump(results_dict["best_params"], f, indent=4)
            
    if "history" in results_dict:
        history_path = os.path.join(save_dir, f"{model_name.lower()}_training_history.json")
        with open(history_path, "w") as f:
            json.dump(results_dict["history"].history, f, indent=4)
            
    # 3. Save the Predictions vs Actuals to CSV (Safely Stitched to 1D)
    # Extract continuous 1D line from the 2D predictions matrix
    preds_1d = stitch_2d_to_1d(results_dict["predictions"], horizon)
    actuals_1d = stitch_2d_to_1d(results_dict["y_test_real"], horizon)
    
    # If the original dataframe index is provided, attempt to align it
    index_vals = getattr(y_test, 'index', range(len(actuals_1d)))
    
    # Trim index if necessary (since slicing drops remainder data)
    if len(index_vals) > len(actuals_1d):
        index_vals = index_vals[:len(actuals_1d)]
    
    preds_df = pd.DataFrame({
        "Actual": actuals_1d,
        "Predicted": preds_1d
        }, index=index_vals)
        
    preds_path = os.path.join(save_dir, f"{model_name.lower()}_predictions.csv")
    preds_df.to_csv(preds_path)
        
    print(f"✅ {model_name} artifacts successfully saved to {save_dir}")