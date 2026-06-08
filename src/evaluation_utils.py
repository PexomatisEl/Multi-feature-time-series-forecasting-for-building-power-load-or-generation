import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import src.config as config
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import spicy.stats as stats


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def stitch_2d_to_1d(array_2d, horizon):

    if len(array_2d.shape) == 1:
        return array_2d
        
    stitched = []
    # Jump by 'horizon' to get non-overlapping 24h windows
    for i in range(0, len(array_2d), horizon):
        stitched.extend(array_2d[i, :])
    return np.array(stitched)


def calculate_metrics(y_true, y_pred, model_name, execution_time, horizon=96, y_train=None):
    y_t = np.array(y_true).flatten()
    y_p = np.array(y_pred).flatten()
    
    # Validation
    assert len(y_t) == len(y_p), f"Shape mismatch: {len(y_t)} vs {len(y_p)}"
    assert len(y_t) > 0, "Empty prediction arrays"
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    # CVRMSE (Coefficient of Variation of RMSE) - ASHRAE Standard
    mean_true = np.mean(y_t)
    cv_rmse = rmse / mean_true if mean_true != 0 else 0
    
    # NMBE (Normalized Mean Bias Error) - ASHRAE Standard
    nmbe = (np.sum(y_t - y_p) / ((y_t.size - 1) * mean_true))
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = np.abs(y_t) + np.abs(y_p)
    epsilon = config.METRICS_EPSILON_SCALE * np.mean(np.abs(y_t))
    smape = 100 * np.mean(2 * np.abs(y_p - y_t) / (denominator + epsilon))
    
    # MASE (Mean Absolute Scaled Error)
    if y_train is None:
        raise ValueError("y_train is required for proper MASE calculation.")
    
    y_train_arr = np.array(y_train).flatten()
    
    if len(y_train_arr) <= horizon:
        raise ValueError(f"y_train too short ({len(y_train_arr)}) for horizon {horizon}")
    
    # 'horizon' safely represents 1 full day of steps across all frequencies
    naive_errors = np.abs(y_train_arr[horizon:] - y_train_arr[:-horizon])
    denominator_mase = np.mean(naive_errors)
    
    if denominator_mase == 0:
        mase = np.nan
        print(f"Zero denominator in MASE for {model_name}")
    else:
        mase = mae / denominator_mase
    
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


def plot_predictions(y_true, y_pred, model_name, horizon, num_steps=None, save_path=None):
    """Plots the non-overlapping Day-Ahead trajectory."""
    y_true_1d = stitch_2d_to_1d(np.array(y_true), horizon)
    y_pred_1d = stitch_2d_to_1d(np.array(y_pred), horizon)
    
    if num_steps is None:
        num_steps = horizon * 4
    
    num_steps = min(num_steps, len(y_true_1d), len(y_pred_1d))
    
    plt.figure(figsize=(20, 5))
    plt.plot(y_true_1d[:num_steps], label='Actual Load', color='black', linewidth=1.5)
    plt.plot(y_pred_1d[:num_steps], label=f'{model_name} Prediction', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f"{model_name} Day-Ahead Performance ({horizon} steps/day)", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Energy Consumption (kW)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_experiment_results(results_dict, model_name, horizon_name, y_test, horizon=96):
    """Saves model artifacts, applying the custom JSON encoder."""
    save_dir = f"../results/{horizon_name}/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    
    if "model" in results_dict:
        model = results_dict["model"]
        if hasattr(model, 'save'):
            model_path = os.path.join(save_dir, f"{model_name.lower()}_model.keras")
            model.save(model_path)
        elif isinstance(model, str): 
            pass 
        else:
            model_path = os.path.join(save_dir, f"{model_name.lower()}_model.joblib")
            joblib.dump(model, model_path)
    
    if "best_params" in results_dict:
        params_path = os.path.join(save_dir, f"{model_name.lower()}_params.json")
        with open(params_path, "w") as f:
            json.dump(results_dict["best_params"], f, indent=4, cls=NpEncoder)
    
    if "history" in results_dict:
        history_path = os.path.join(save_dir, f"{model_name.lower()}_training_history.json")
        with open(history_path, "w") as f:
            json.dump(results_dict["history"].history, f, indent=4, cls=NpEncoder)
    
    preds_1d = stitch_2d_to_1d(results_dict["predictions"], horizon)
    actuals_1d = stitch_2d_to_1d(results_dict["y_test_real"], horizon)
    
    index_vals = getattr(y_test, 'index', range(len(actuals_1d)))
    if len(index_vals) > len(actuals_1d):
        index_vals = index_vals[:len(actuals_1d)]
    
    preds_df = pd.DataFrame({"Actual": actuals_1d, "Predicted": preds_1d}, index=index_vals)
    preds_path = os.path.join(save_dir, f"{model_name.lower()}_predictions.csv")
    preds_df.to_csv(preds_path)
    
    print(f"{model_name} artifacts successfully saved to {save_dir}")


def create_metrics_summary_table(all_results_dict):
    """Creates a summary table."""
    rows = []
    for model_name, metrics in all_results_dict.items():
        row = {
            "Model": metrics.get("Model", model_name),
            "RMSE": f"{metrics['RMSE']:.4f}",
            "MAE": f"{metrics['MAE']:.4f}",
            "R²": f"{metrics['R2']:.4f}",
            "CVRMSE (%)": f"{metrics['CVRMSE']*100:.2f}",
            "NMBE": f"{metrics['NMBE']:.6f}",
            "sMAPE (%)": f"{metrics['sMAPE']:.2f}",
            "MASE": f"{metrics['MASE']:.4f}" if not np.isnan(metrics['MASE']) else "N/A",
            "Time (s)": f"{metrics['Time (s)']:.2f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("RMSE", ascending=True)
    return df

def plot_hourly_error_heatmap(model_results_dict):
    """
    Generates an Hour-of-Day Error Heatmap.
    Expects predictions and targets of shape (samples, 96).
    """
    print("\n Generating Master Hourly Error Heatmap")
    
    heatmap_data = {}
    
    for model_name, results in model_results_dict.items():
        # Extract true values and predictions
        y_true = results['y_test_real']
        y_pred = results['predictions']
        
        # Calculate Absolute Error for every single step across all test samples
        absolute_errors = np.abs(y_true - y_pred)
        
        # Average the errors across all samples to get the mean error per step (Shape: 96)
        mean_step_errors = np.mean(absolute_errors, axis=0)
        
        # Group the 96 15-minute intervals into 24 hours (averaging every 4 steps)
        # This can be dynamically adjusted if the horizon changes
        steps_per_hour = y_true.shape[1] // 24
        hourly_errors = [np.mean(mean_step_errors[i*steps_per_hour : (i+1)*steps_per_hour]) for i in range(24)]
        
        # Store in our dictionary
        heatmap_data[model_name] = hourly_errors

    # Convert to DataFrame for Seaborn (Rows = Models, Columns = Hours)
    error_df = pd.DataFrame(heatmap_data).T
    error_df.columns = [f"{i:02d}:00" for i in range(24)]
    

    # Plotting
    
    plt.figure(figsize=(16, 4 + len(model_results_dict) * 0.8))
    
    # Colormap: White/Light Yellow (Low Error) to Dark Red (High Error)
    ax = sns.heatmap(error_df, cmap="YlOrRd", annot=True, fmt=".2f", 
                     linewidths=.5, cbar_kws={'label': 'Mean Absolute Error (MAE)'})
    
    plt.title('Hour-of-Day Forecasting Error (MAE) by Model Architecture', fontsize=16, pad=15)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Rotate y-axis labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return error_df

def plot_uncertainty_bounds(chronos_results_dict, sample_idx=0, horizon=96):
    """
    Plots the actual load, the median prediction, and the 80% confidence interval.
    """
    print("\n Generating Uncertainty Bounds Plot for Chronos")
    
    # Extract the specific 96-step window we want to look at
    y_true = chronos_results_dict['y_test_real'][sample_idx][:horizon]
    y_pred = chronos_results_dict['predictions'][sample_idx][:horizon]
    y_lower = chronos_results_dict['lower_bounds'][sample_idx][:horizon]
    y_upper = chronos_results_dict['upper_bounds'][sample_idx][:horizon]
    
    plt.figure(figsize=(16, 6))
    
    # Plot the Actuals and the Median
    plt.plot(y_true, color='black', label='Actual Load', linewidth=2)
    plt.plot(y_pred, color='red', linestyle='--', label='Chronos Median Forecast (P50)')
    
    # The Magic: Fill the area between the 10th and 90th percentile bounds
    plt.fill_between(range(horizon), y_lower, y_upper, color='red', alpha=0.15, 
                     label='80% Confidence Interval (P10 - P90)')
    
    plt.title('Chronos Day-Ahead Forecast with Uncertainty Bounds', fontsize=16)
    plt.xlabel('Time Steps (15-min intervals)', fontsize=12)
    plt.ylabel('Energy Consumption (kW)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    
    # Add a subtle grid
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def run_all_residual_diagnostics(y_true, y_pred, model_name, horizon_idx=0):
    """
    y_true/y_pred: Expected shape (samples, horizon)
    horizon_idx: Which time-step to analyze (0 = 15-min ahead)
    """
    # Isolate the specific horizon step for analysis
    target_residuals = y_true[:, horizon_idx] - y_pred[:, horizon_idx]
    
    print(f"\n--- DIAGNOSTICS FOR {model_name.upper()} (Step: {horizon_idx}) ---")
    
    # 1. ACF/PACF
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    plot_acf(target_residuals, lags=48, ax=ax[0], title=f"{model_name} ACF")
    plot_pacf(target_residuals, lags=48, ax=ax[1], title=f"{model_name} PACF")
    plt.tight_layout()
    plt.show()
    
    # 2. Shapiro-Wilk
    stat, p = stats.shapiro(target_residuals)
    print(f"Shapiro-Wilk P-Value: {p:.4e}")
    
    # 3. Mean Bias by Hour
    # Create a DataFrame for grouping
    res_df = pd.DataFrame({'res': target_residuals}, index=df_test.index[:len(target_residuals)])
    hourly = res_df.groupby(res_df.index.hour).mean()
    
    plt.figure(figsize=(8, 3))
    sns.barplot(x=hourly.index, y=hourly['res'], color='salmon')
    plt.axhline(0, color='black', lw=1)
    plt.title(f"Mean Residual Bias by Hour ({model_name})")
    plt.show()

def evaluate_statistical_significance(y_true, y_pred_a, y_pred_b, model_a_name, model_b_name):
    """
    Performs a Paired t-test on the Absolute Errors of two models.
    Assumes arrays are of the same shape.
    """
    # Flatten arrays to compare all individual timestep errors
    errors_a = np.abs(np.array(y_true).flatten() - np.array(y_pred_a).flatten())
    errors_b = np.abs(np.array(y_true).flatten() - np.array(y_pred_b).flatten())
    
    # Perform Paired t-test
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
    
    print("STATISTICAL SIGNIFICANCE TEST (Paired t-test)")
    print(f"   {model_a_name} vs {model_b_name}")
    
    mean_err_a = np.mean(errors_a)
    mean_err_b = np.mean(errors_b)
    
    print(f"Mean Absolute Error ({model_a_name}): {mean_err_a:.4f}")
    print(f"Mean Absolute Error ({model_b_name}): {mean_err_b:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_value:.4e}")
    
    alpha = 0.05
    if p_value < alpha:
        winner = model_a_name if mean_err_a < mean_err_b else model_b_name
        print(f"CONCLUSION: The difference IS statistically significant (p < {alpha}).")
        print(f"{winner} is mathematically superior.")
    else:
        print(f"CONCLUSION: The difference is NOT statistically significant (p >= {alpha}).")
        print(" Both models perform practically the same within statistical variance.")