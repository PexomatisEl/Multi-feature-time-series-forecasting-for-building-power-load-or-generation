import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from chronos import ChronosPipeline, Chronos2Pipeline
import torch
from src.evaluation_utils import calculate_metrics, plot_predictions
from src.model_prep_tools import create_multivariate_sequences
import src.config as config
import os
from tqdm.auto import tqdm

############################ LSTM ARCHITECTURE ############################

def run_power_lstm(X_train, y_train, X_val, y_val, X_test, y_test, 
                   horizon_name, horizon, epochs, batch_size, random_state, 
                   experiment_name="Power_LSTM",
                   target_scaler=None, train_from_scratch=False):
    
    lstm_cfg = config.LSTM_CONFIG
    train_cfg = config.TRAINING_PARAMS
    
    
    print(f"\n RUNNING {experiment_name.upper()} ({horizon_name}) | Horizon: {horizon}")
    start_time = time.time()
    
    cache_dir = f"../results/{horizon_name}/{experiment_name}"
    cache_path = os.path.join(cache_dir, "predictions.npz")
    model_save_path = os.path.join(cache_dir, f"{experiment_name.lower()}_model.keras")
    
    # Bypass everything if predictions are already cached
    if not train_from_scratch and os.path.exists(cache_path):
        print(f"Cache found! Loading cached inferences from {cache_path}")
        loaded = np.load(cache_path)
        
        # Load model purely for artifact saving purposes if needed
        model = load_model(model_save_path) if os.path.exists(model_save_path) else "Cached_LSTM"
        
        return {
            'model': model,
            'predictions': loaded['predictions'],
            'y_test_real': loaded['y_test_real'],
            'history': None,
            'execution_time': loaded['execution_time'].item()
        }
    
    # Clear memory and set seeds
    tf.keras.backend.clear_session()
    tf.random.set_seed(random_state)
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        
        Bidirectional(LSTM(lstm_cfg["hidden_units"][0], return_sequences=True)),
        LayerNormalization(), 
        Dropout(lstm_cfg["dropout"]),
        
        LSTM(lstm_cfg["hidden_units"][1], return_sequences=True),
        LayerNormalization(), 
        Dropout(lstm_cfg["dropout"]),
        
        GlobalAveragePooling1D(),
        
        Dense(lstm_cfg["hidden_units"][0], activation='relu'), 
        Dense(horizon, activation='linear') 
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lstm_cfg["learning_rate"])
    if lstm_cfg.get("loss") == "mse":
        loss_fn = 'mse'
    else:
        loss_fn = tf.keras.losses.Huber(delta=lstm_cfg.get("huber_delta", 0.05))
        
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    history = None
    
    # Bypass training if weights exist but npz does not
    if not train_from_scratch and os.path.exists(model_save_path):
        print(f"Loading pre-trained weights from {model_save_path}")
        model = load_model(model_save_path)
    else:
        print(f"Training LSTM for {horizon} steps")
        early_stop = EarlyStopping(monitor='val_loss', patience=train_cfg["early_stopping_patience"], restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=train_cfg["lr_reduce_factor"], 
                                         patience=train_cfg["lr_reduce_patience"], min_lr=train_cfg["min_learning_rate"], verbose=1)
        
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val, y_val), 
            callbacks=[early_stop, lr_scheduler],
            verbose=1 
        )
    
    print("Generating predictions...")
    test_preds = model.predict(X_test, verbose=0)
    execution_time = time.time() - start_time
    
    if target_scaler is not None:
        if len(test_preds.shape) == 1:
            test_preds = test_preds.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
        else:
            y_test_reshaped = y_test
            
        test_preds = target_scaler.inverse_transform(test_preds)
        y_test_unscaled = target_scaler.inverse_transform(y_test_reshaped)
    else:
        y_test_unscaled = y_test
        
    # Save cache for instant loading next time
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, predictions=test_preds, y_test_real=y_test_unscaled, 
             execution_time=np.array(execution_time))
    print(f"Inference successfully cached to {cache_path}")
    
    return {
        'model': model,
        'predictions': test_preds,
        'y_test_real': y_test_unscaled,
        'history': history,
        'execution_time': execution_time
    }
################# Vanila Transformer Architecture #################

def run_vanilla_transformer(X_train, y_train, X_val, y_val, X_test, y_test, 
                            horizon_name, horizon, epochs, batch_size, random_state, 
                            experiment_name="Vanilla_Transformer", # <--- UPGRADED
                            target_scaler=None, train_from_scratch=False):
    
    trans_cfg = config.TRANSFORMER_CONFIG
    train_cfg = config.TRAINING_PARAMS
    
    print(f"\nRUNNING {experiment_name.upper()} ({horizon_name}) | Horizon: {horizon}")
    start_time = time.time()
    
    # Cached model
    cache_dir = f"../results/{horizon_name}/{experiment_name}"
    cache_path = os.path.join(cache_dir, "predictions.npz")
    model_save_path = os.path.join(cache_dir, f"{experiment_name.lower()}_model.keras")
    
    if not train_from_scratch and os.path.exists(cache_path):
        print(f"Loading cached inferences from {cache_path}")
        loaded = np.load(cache_path)
        model = load_model(model_save_path) if os.path.exists(model_save_path) else "Cached_Transformer"
        return {
            'model': model,
            'predictions': loaded['predictions'],
            'y_test_real': loaded['y_test_real'],
            'history': None,
            'execution_time': loaded['execution_time'].item()
        }

    seq_length = X_train.shape[1]
    num_features = X_train.shape[2]
    d_model = trans_cfg["d_model"]
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(random_state)
    
    # Transformer Architecture
    inputs = layers.Input(shape=(seq_length, num_features))
    x = layers.Dense(d_model)(inputs)
    
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    pos_emb = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    x = x + pos_emb
    
    attn_output = layers.MultiHeadAttention(num_heads=trans_cfg["num_heads"], key_dim=d_model)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn = layers.Dense(trans_cfg["ffn_dim"], activation='relu')(x)
    ffn = layers.Dropout(trans_cfg["dropout"])(ffn)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(trans_cfg["dropout"])(x)
    
    head_dim = trans_cfg.get("head_dim", 256)
    x = layers.Dense(head_dim, activation='relu')(x)
    outputs = layers.Dense(horizon, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=trans_cfg["learning_rate"])
    
    if trans_cfg.get("loss") == "mse":
        loss_fn = 'mse'
    else:
        loss_fn = tf.keras.losses.Huber(delta=trans_cfg.get("huber_delta", 0.05))
        
    model.compile(optimizer=optimizer, loss=loss_fn)

    history = None
    
    if not train_from_scratch and os.path.exists(model_save_path):
        print(f"Loading pre-trained weights from {model_save_path}")
        model = load_model(model_save_path)
    else:
        print(f"Training Vanilla Transformer for {horizon} steps")
        early_stop = EarlyStopping(monitor='val_loss', patience=train_cfg["early_stopping_patience"], restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=train_cfg["lr_reduce_factor"], 
                                         patience=train_cfg["lr_reduce_patience"], min_lr=train_cfg["min_learning_rate"], verbose=1)

        history = model.fit(
            X_train, y_train, 
            epochs=epochs, batch_size=batch_size, 
            validation_data=(X_val, y_val),  
            callbacks=[early_stop, lr_scheduler], verbose=1 
        )

    print("Generating predictions")
    test_preds = model.predict(X_test, verbose=0)
    execution_time = time.time() - start_time
    
    if target_scaler is not None:
        if len(test_preds.shape) == 1:
            test_preds = test_preds.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
        else:
            y_test_reshaped = y_test
            
        test_preds = target_scaler.inverse_transform(test_preds)
        y_test_unscaled = target_scaler.inverse_transform(y_test_reshaped)
    else:
        y_test_unscaled = y_test
    
    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, predictions=test_preds, y_test_real=y_test_unscaled, 
             execution_time=np.array(execution_time))
    
    return {
        'model': model,
        'predictions': test_preds,
        'y_test_real': y_test_unscaled,
        'history': history,
        'execution_time': execution_time
    }

##################### PatchTST Architecture - Univariate ######################

def run_keras_patchtst(X_train, y_train, X_val, y_val, X_test, y_test, 
                       horizon_name, horizon, epochs, batch_size, 
                       patch_len, stride, target_idx, random_state,
                       experiment_name="PatchTST",
                       target_scaler=None, train_from_scratch=False):
    
    patch_cfg = config.PATCHTST_CONFIG
    train_cfg = config.TRAINING_PARAMS
    
    print(f"\nRUNNING {experiment_name.upper()} ({horizon_name}) | Horizon: {horizon}")
    start_time = time.time()
    
    cache_dir = f"../results/{horizon_name}/{experiment_name}"
    cache_path = os.path.join(cache_dir, "predictions.npz")
    model_save_path = os.path.join(cache_dir, f"{experiment_name.lower()}_model.keras")
    
    # BUILD THE ARCHITECTURE
    seq_length = X_train.shape[1]
    num_features = X_train.shape[2]
    d_model = patch_cfg["d_model"]

    tf.keras.backend.clear_session()
    tf.random.set_seed(random_state)
    
    inputs = layers.Input(shape=(seq_length, num_features))
    x_ci = layers.Lambda(lambda z: tf.reshape(tf.transpose(z, perm=[0, 2, 1]), [-1, seq_length, 1]))(inputs)
    x_patched = layers.Conv1D(filters=d_model, kernel_size=patch_len, 
                              strides=stride, padding='valid')(x_ci)
    
    num_patches = (seq_length - patch_len) // stride + 1
    position = np.arange(num_patches)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((num_patches, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    pos_emb = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    x = x_patched + pos_emb
    
    attn_output = layers.MultiHeadAttention(num_heads=patch_cfg["num_heads"], key_dim=d_model)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn = layers.Dense(patch_cfg["ffn_dim"], activation='relu')(x)
    ffn = layers.Dropout(patch_cfg["dropout"])(ffn)
    ffn = layers.Dense(d_model)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(patch_cfg["dropout"])(x)
    x = layers.Dense(horizon, activation='linear')(x)
    outputs = layers.Lambda(lambda z: tf.reshape(z, [-1, num_features, horizon])[:, target_idx, :])(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=patch_cfg["learning_rate"])
    
    if patch_cfg.get("loss") == "mse":
        loss_fn = 'mse'
    else:
        loss_fn = tf.keras.losses.Huber(delta=patch_cfg.get("huber_delta", 0.05))
        
    model.compile(optimizer=optimizer, loss=loss_fn)

    # CHECK CACHE (Inject weights directly into the built model)
    if not train_from_scratch and os.path.exists(cache_path):
        print(f"Loading cached inferences from {cache_path}")
        loaded = np.load(cache_path)
        if os.path.exists(model_save_path):
            model.load_weights(model_save_path)
        return {
            'model': model,
            'predictions': loaded['predictions'],
            'y_test_real': loaded['y_test_real'],
            'history': None,
            'execution_time': loaded['execution_time'].item()
        }

    history = None
    
    # TRAINING OR LOADING (If cache is missing)
    if not train_from_scratch and os.path.exists(model_save_path):
        print(f"Loading pre-trained weights from {model_save_path}")
        model.load_weights(model_save_path)
    else:
        print(f"Training PatchTST for {horizon} steps")
        early_stop = EarlyStopping(monitor='val_loss', patience=train_cfg["early_stopping_patience"], restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=train_cfg["lr_reduce_factor"], 
                                         patience=train_cfg["lr_reduce_patience"], min_lr=train_cfg["min_learning_rate"], verbose=1)

        history = model.fit(
            X_train, y_train, 
            epochs=epochs, batch_size=batch_size, 
            validation_data=(X_val, y_val), 
            callbacks=[early_stop, lr_scheduler], verbose=1 
        )
        # save model to make sure weights generate
        os.makedirs(cache_dir, exist_ok=True)
        model.save(model_save_path)

    print("Generating predictions")
    test_preds = model.predict(X_test, verbose=0)
    execution_time = time.time() - start_time
    
    if target_scaler is not None:
        if len(test_preds.shape) == 1:
            test_preds = test_preds.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
        else:
            y_test_reshaped = y_test
            
        test_preds = target_scaler.inverse_transform(test_preds)
        y_test_unscaled = target_scaler.inverse_transform(y_test_reshaped)
    else:
        y_test_unscaled = y_test
    
    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, predictions=test_preds, y_test_real=y_test_unscaled,
             execution_time=np.array(execution_time))
    
    return {
        'model': model,
        'predictions': test_preds,
        'y_test_real': y_test_unscaled,
        'history': history,
        'execution_time': execution_time
    }


# Create a dummy history class to prevent the JSON saver from crashing
class DummyHistory:
    def __init__(self):
        self.history = {"loss": [0], "val_loss": [0], "note": "Zero-shot model, no training history."}

def run_chronos_univariate(df, target_col='energy_consumption', horizon_name="15_min",
                           model_name="amazon/chronos-t5-small", batch_size=16, 
                           context_length=512, experiment_name=None,
                           train_from_scratch=False):
    
    horizon = config.HORIZON_STEPS[horizon_name]
    clean_model_name = model_name.replace("/", "_")
    
    # Use custom experiment name if provided
    exp_dir_name = experiment_name if experiment_name else clean_model_name
    
    print(f"\n RUNNING CHRONOS ZERO-SHOT ({exp_dir_name}) | Horizon: {horizon}")
    
    start_time = time.time()
    
    # Extract Raw Unscaled Data
    y_all = df[target_col].values
    train_size = int(len(y_all) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    y_train_raw = y_all[:train_size] 

    # Cache uses exp_dir_name so context ablations don't overwrite
    cache_dir = f"../results/{horizon_name}/{exp_dir_name}"
    cache_path = os.path.join(cache_dir, "predictions.npz")
    
    # Load Cached Predictions
    if not train_from_scratch and os.path.exists(cache_path):
        print(f"Cache found! Loading cached inferences from {cache_path}")
        loaded = np.load(cache_path)
        predictions = loaded['predictions']
        lower_bounds = loaded['lower_bounds']
        upper_bounds = loaded['upper_bounds']
        y_test_real = loaded['y_test_real']
        execution_time = loaded['execution_time'].item()
        
    else:
        # Full Inference Pipeline
        print(f"Loading {model_name}")
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Flush VRAM before loading the pipeline
            
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        context_list = []
        y_test_real_seq = []
        
        print("Slicing sliding windows for test set...")
        for i in range(train_size, len(y_all) - horizon + 1):
            start_idx = max(0, i - context_length) 
            ctx = y_all[start_idx:i] 
            context_list.append(torch.tensor(ctx, dtype=torch.float32))
            y_test_real_seq.append(y_all[i : i + horizon])

        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        print(f"Running batch inference (Batch Size: {batch_size})")
        with torch.no_grad():
            # THE TQDM PROGRESS BAR LOOP
            for b in tqdm(range(0, len(context_list), batch_size), desc=f"Predicting {clean_model_name}"):
                batch_ctx = context_list[b : b + batch_size]
                forecast = pipeline.predict(batch_ctx, prediction_length=horizon)
                
                # Extract the 10th, 50th (median), and 90th percentiles
                forecast_np = forecast.cpu().numpy()
                median_forecast = np.quantile(forecast_np, 0.5, axis=1)
                lower_forecast = np.quantile(forecast_np, 0.1, axis=1)
                upper_forecast = np.quantile(forecast_np, 0.9, axis=1)
                
                predictions.extend(median_forecast)
                lower_bounds.extend(lower_forecast)
                upper_bounds.extend(upper_forecast)

        predictions = np.array(predictions)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        y_test_real = np.array(y_test_real_seq)
        execution_time = time.time() - start_time
        
        # Save Cache
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, predictions=predictions, lower_bounds=lower_bounds, 
                 upper_bounds=upper_bounds, y_test_real=y_test_real, 
                 execution_time=np.array(execution_time))
        print(f"Inference successfully cached to {cache_path}")

    # Evaluation and Plotting
    metrics_dict = calculate_metrics(y_test_real, predictions, clean_model_name, execution_time, 
                                     horizon=horizon, y_train=y_train_raw)
    
    plot_predictions(y_test_real, predictions, clean_model_name, horizon, num_steps=horizon * 7)
    
    return {
        "model": clean_model_name,
        "predictions": predictions,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "y_test_real": y_test_real,
        "history": DummyHistory(), 
        "metrics": metrics_dict
    }


def run_chronos2_multivariate(df, target_col='energy_consumption', horizon_name="15_min",
                              model_name=config.CHRONOS_2_CONFIG["model_name"], 
                              batch_size=config.CHRONOS_2_CONFIG["batch_size"], 
                              context_length=config.CHRONOS_2_CONFIG["context_length"]):
    horizon = config.HORIZON_STEPS[horizon_name]
    start_time = time.time()
    
    df_chronos = df.copy()
    if not isinstance(df_chronos.index, pd.DatetimeIndex):
        df_chronos.index = pd.to_datetime(df_chronos.index)
        
    df_chronos = df_chronos.reset_index(names=['timestamp'])
    covariate_cols = [c for c in df_chronos.columns if c not in ['timestamp', 'id', target_col]]
    
    train_size = int(len(df_chronos) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    test_data = df_chronos.iloc[train_size:].copy()
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)
    
    predictions = []
    lower_bounds = []
    upper_bounds = []
    y_test_real_seq = []
    num_windows = len(test_data) - horizon + 1

    for b_start in range(0, num_windows, batch_size):
        b_end = min(b_start + batch_size, num_windows)
        ctx_list, fut_list = [], []
        
        for i in range(b_start, b_end):
            start_idx = max(0, train_size + i - context_length)
            ctx_df = df_chronos.iloc[start_idx : train_size + i].copy()
            ctx_df['id'] = f"window_{i}" 
            
            fut_df = df_chronos.iloc[train_size + i : train_size + i + horizon].copy()
            fut_df['id'] = f"window_{i}"
            
            ctx_list.append(ctx_df)
            fut_list.append(fut_df)
            y_test_real_seq.append(fut_df[target_col].values)

        batch_context_df = pd.concat(ctx_list, ignore_index=True)
        batch_future_df = pd.concat(fut_list, ignore_index=True)
        
        pred_df = pipeline.predict_df(
            batch_context_df,
            future_df=batch_future_df[['id', 'timestamp'] + covariate_cols],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target=target_col,
            batch_size=batch_size 
        )
        
        for i in range(b_start, b_end):
            # Extract all three percentiles
            window_preds = pred_df[pred_df['id'] == f"window_{i}"]['0.5'].values
            window_lower = pred_df[pred_df['id'] == f"window_{i}"]['0.1'].values
            window_upper = pred_df[pred_df['id'] == f"window_{i}"]['0.9'].values
            
            predictions.append(window_preds)
            lower_bounds.append(window_lower)
            upper_bounds.append(window_upper)
            
        if b_start % (batch_size * 10) == 0 and b_start > 0:
            print(f"Evaluated {b_start}/{num_windows} test windows...")

    predictions = np.array(predictions)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    y_test_real = np.array(y_test_real_seq)
    execution_time = time.time() - start_time
    
    return {
        'model': "Chronos_Multivariate",
        'predictions': predictions,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'y_test_real': y_test_real,
        'history': DummyHistory(),
        'execution_time': execution_time
    }


def run_chronos2_multivariate_v2(df, deterministic_covariates, experiment_name, 
                                 target_col='energy_consumption', horizon_name="15_min", 
                                 model_name="autogluon/chronos-2-small", batch_size=32, 
                                 context_length=512, train_from_scratch=True):
    
    horizon = config.HORIZON_STEPS[horizon_name]
    start_time = time.time()
    
    print(f"\n RUNNING {experiment_name.upper()} | Horizon: {horizon}")
    
    cache_dir = f"../results/{horizon_name}/{experiment_name}"
    cache_path = os.path.join(cache_dir, "predictions.npz")
    
    # Bypass Inference if Cache Exists
    if not train_from_scratch and os.path.exists(cache_path):
        print(f"Loading cached inferences from {cache_path}")
        loaded = np.load(cache_path)
        return {
            'model': experiment_name,
            'predictions': loaded['predictions'],
            'lower_bounds': loaded['lower_bounds'],
            'upper_bounds': loaded['upper_bounds'],
            'y_test_real': loaded['y_test_real'],
            'history': DummyHistory(),
            'execution_time': loaded['execution_time'].item()
        }

    # Full Inference pipeline
    df_chronos = df.copy()
    if not isinstance(df_chronos.index, pd.DatetimeIndex):
        df_chronos.index = pd.to_datetime(df_chronos.index)
        
    df_chronos = df_chronos.reset_index(names=['timestamp'])
    
    train_size = int(len(df_chronos) * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    test_data = df_chronos.iloc[train_size:].copy()
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear cuda memory before loading a new model size
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    pipeline = Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)
    
    predictions, lower_bounds, upper_bounds, y_test_real_seq = [], [], [], []
    num_windows = len(test_data) - horizon + 1

    for b_start in tqdm(range(0, num_windows, batch_size), desc=f"Predicting {experiment_name}"):
        b_end = min(b_start + batch_size, num_windows)
        ctx_list, fut_list = [], []
        
        for i in range(b_start, b_end):
            start_idx = max(0, train_size + i - context_length)
            
            # Context gets all features
            ctx_df = df_chronos.iloc[start_idx : train_size + i].copy()
            ctx_df['id'] = f"window_{i}" 
            
            # Future only gets deterministic features
            fut_df = df_chronos.iloc[train_size + i : train_size + i + horizon].copy()
            fut_df['id'] = f"window_{i}"
            
            ctx_list.append(ctx_df)
            fut_list.append(fut_df)
            y_test_real_seq.append(fut_df[target_col].values)

        batch_context_df = pd.concat(ctx_list, ignore_index=True)
        batch_future_df = pd.concat(fut_list, ignore_index=True)
        
        # Prediction
        pred_df = pipeline.predict_df(
            batch_context_df,
            future_df=batch_future_df[['id', 'timestamp'] + deterministic_covariates],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target=target_col,
            batch_size=batch_size 
        )
        
        for i in range(b_start, b_end):
            window_preds = pred_df[pred_df['id'] == f"window_{i}"]['0.5'].values
            window_lower = pred_df[pred_df['id'] == f"window_{i}"]['0.1'].values
            window_upper = pred_df[pred_df['id'] == f"window_{i}"]['0.9'].values
            
            predictions.append(window_preds)
            lower_bounds.append(window_lower)
            upper_bounds.append(window_upper)

    execution_time = time.time() - start_time
    
    predictions = np.array(predictions)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    y_test_real = np.array(y_test_real_seq)
    
    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, predictions=predictions, lower_bounds=lower_bounds, 
             upper_bounds=upper_bounds, y_test_real=y_test_real, 
             execution_time=np.array(execution_time))
    print(f"Inference successfully cached to {cache_path}")
    
    return {
        'model': experiment_name,
        'predictions': predictions,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'y_test_real': y_test_real,
        'history': DummyHistory(),
        'execution_time': execution_time
    }