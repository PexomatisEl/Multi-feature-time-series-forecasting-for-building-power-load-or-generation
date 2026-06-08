import pandas as pd
import numpy as np
import src.config as config
import holidays

class ThesisDataPipeline:
    """
    A data preprocessing pipeline.
    applies causal imputation, and then resamples to the target horizon.
    """
    def __init__(self, target_freq_minutes):
        self.target_freq = target_freq_minutes
        self.target_str = f'{target_freq_minutes}min'
        
        # Native resolution of the raw data
        self.native_freq = 15 
        self.native_steps_per_hour = int(60 / self.native_freq) # Always 4


    def cross_sensor_patch(self, df):
        df_clean = df.copy()
        pairs = [
            ('temperature_v1', 'temperature_v2'),
            ('relative_humidity_v1', 'relative_humidity_v2'),
            ('wind_speed_v1', 'wind_speed_v2'),
            ('clouds_v1', 'clouds_v2')
        ]
        
        print("\n Cross Sensor Patching and Validation")

        # Convert wind_speed_v2 from m/s to km/h before patching
        if 'wind_speed_v2' in df_clean.columns:
            df_clean['wind_speed_v2'] = df_clean['wind_speed_v2'] * 3.6
            print(" [Unit Alignment]: Converted wind_speed_v2 from m/s to km/h")

        for primary, backup in pairs:
            if primary in df_clean.columns and backup in df_clean.columns:
                
                # Validation: Calculate correlation before patching
                # We drop NaNs temporarily just to calculate the correlation of the overlap
                valid_overlap = df_clean.dropna(subset=[primary, backup])
                if len(valid_overlap) > 0:
                    corr_spearman = valid_overlap[primary].corr(valid_overlap[backup], method='spearman')
                    print(f"[{primary} & {backup}] Spearman Corr: {corr_spearman:.4f}")
                    
                    if corr_spearman < 0.95:
                        print(f" {primary} and {backup} deviate significantly. So there might be a data quality issue.")

                # Execute the Patch
                missing_before = df_clean[primary].isna().sum()
                df_clean[primary] = df_clean[primary].fillna(df_clean[backup])
                missing_after = df_clean[primary].isna().sum()
                salvaged = missing_before - missing_after
                
                if salvaged > 0:
                    print(f"Salvaged {salvaged} rows using {backup}.")
                
                # Clean up redundant for training purposes features
                df_clean = df_clean.drop(columns=[backup])
                
        return df_clean

    def temporal_bin_imputation(self, df):
        print(f"Causal Imputation Cascade at native {self.native_freq}-min resolution")
        df_clean = df.copy()
        
        # FFill Price (SMP)
        if 'smp' in df_clean.columns:
            df_clean['smp'] = df_clean['smp'].ffill()

        # BIN 1: Small Gaps (<= 2 hours) -> Linear Interpolation

        bin1_hours = config.IMPUTATION_BINS["bin1_threshold_hours"]
        limit_steps = int(self.native_steps_per_hour * bin1_hours)
        df_clean = df_clean.interpolate(method='linear', limit=limit_steps)
        

        # BIN 2: Medium Gaps (<= 1 week) -> Strictly Homologous Causal Blend
        # Only blends multiples of 1 week to preserve working environment shift schedules

        steps_168h = int(168 * self.native_steps_per_hour)
        steps_336h = int(336 * self.native_steps_per_hour)
        
        def safe_blend(sig1, sig2):
            return (sig1.fillna(sig2) + sig2.fillna(sig1)) / 2

        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                sig_168h = df_clean[col].shift(steps_168h)
                sig_336h = df_clean[col].shift(steps_336h)
                
                final_patch = safe_blend(sig_168h, sig_336h)
                df_clean[col] = df_clean[col].fillna(final_patch)


        # BIN 3: Large Gaps (> 1 week) -> Strictly Causal Macro-Patch

        steps_52_weeks = int(52 * 7 * 24 * self.native_steps_per_hour)
        steps_4_weeks = int(4 * 7 * 24 * self.native_steps_per_hour) 
        
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                patch_52w = df_clean[col].shift(steps_52_weeks)
                patch_4w = df_clean[col].shift(steps_4_weeks)
                df_clean[col] = df_clean[col].fillna(patch_52w).fillna(patch_4w)
                
        # Final Truncation of Unsalvageable Initial Boundaries
        df_clean = df_clean.dropna()
        print(f"15-min Master Dataset is 100% continuous. Causal Start Date: {df_clean.index.min()}")
            
        return df_clean
    
    

    def resample_and_engineer(self, df, target_cols=['energy_consumption', 'energy_generation', 'smp', 'temperature_v1']):
            """
            Enforces a 24-hour Operational Lead Time.
            Information Gap: Exactly 24 hours for all frequencies.
            """

            print(f"Resampling to {self.target_str} with 24h Action Lead Time")
            
            # Basic Resampling
            df_resampled = df.resample(self.target_str).mean()
            
            # Dynamic Lead Time Calculation
            week_steps = int(10080 / self.target_freq) # 7 days
            
            for col in target_cols:
                if col in df_resampled.columns:
                    # Value at T - 1 week (Highly predictive for weekly seasonality)
                    df_resampled[f'{col}_lag_1w'] = df_resampled[col].shift(week_steps)
                    
            
            # Cyclical Time Features
            
            # Daily cycle (0 to 24 hours)
            time_decimal = df_resampled.index.hour + (df_resampled.index.minute / 60.0)
            df_resampled['hour_sin'] = np.sin(2 * np.pi * time_decimal / 24.0)
            df_resampled['hour_cos'] = np.cos(2 * np.pi * time_decimal / 24.0)
            
            # Weekly cycle (0 = Monday, 6 = Sunday)
            df_resampled['day_sin'] = np.sin(2 * np.pi * df_resampled.index.dayofweek / 7.0)
            df_resampled['day_cos'] = np.cos(2 * np.pi * df_resampled.index.dayofweek / 7.0)
            
            # Yearly cycle (1 to 12 months)
            df_resampled['month_sin'] = np.sin(2 * np.pi * df_resampled.index.month / 12.0)
            df_resampled['month_cos'] = np.cos(2 * np.pi * df_resampled.index.month / 12.0)

            # Feature Engineering
            # 1. Temperature times Hour Interaction
            if 'temperature_v1' in df_resampled.columns:
                df_resampled['temp_x_hour'] = df_resampled['temperature_v1']*df_resampled['hour_sin']
            

            # 2. Solar PV Profile (Sun angle modified by clouds)
            if 'clouds_v1' in df_resampled.columns:
                df['solar_pv_profile'] = df_resampled['hour_cos']*(1-(df_resampled['clouds_v1']/100))
            # Categorical flags
            
            # Generate the Greek holiday dictionary dynamically based on the dataframe's years
            years_in_data = df_resampled.index.year.unique().tolist()
            gr_holidays = holidays.country_holidays('GR', years=years_in_data)
            
            df_resampled['is_holiday'] = df_resampled.index.map(lambda x: x in gr_holidays).astype(int)
            df_resampled['is_weekend'] = (df_resampled.index.weekday >= 5).astype(int)
            df_resampled['is_non_working_day'] = ((df_resampled['is_weekend'] == 1) | (df_resampled['is_holiday'] == 1)).astype(int)
            
            # Final Cleanup
            # We drop NaNs created by the 1-week shift
            df_resampled = df_resampled.dropna()
            
            return df_resampled
    
    def force_native_grid(self, load, pv, smp, weather, battery=None):
        """
        Base Alignment & Exposing Invisible Gaps.
        Forces the raw data onto a strict 15-minute clock.
        """
        print(f"Aligning datasets and forcing strict {self.native_freq}-min grid")
            
        # FFill SMP to 1-min resolution first so it joins flawlessly
        smp_dense = smp.resample('1min').ffill()
            
        # Create the list of dataframes to join
        join_list = [pv, smp_dense, weather]
        if battery is not None:
            join_list.append(battery)
                
        # Outer join to gather all timestamps
        df_raw = load.join(join_list, how="outer").sort_index()
            
        # Remove impossible energy outliers
        if "energy_consumption" in df_raw.columns:
            df_raw = df_raw[df_raw["energy_consumption"] < 1000]
            
        # Force strict native grid to expose outages
        df_grid = df_raw.resample(f'{self.native_freq}min').asfreq()
        
        return df_grid
    

# Utility Functions for Modeling

def create_multivariate_sequences(data_array, target_col_idx, seq_length, horizon):
    """
    Creates sequences for Direct Multi-Step Forecasting (MIMO).
    X shape: (samples, seq_length, features)
    y shape: (samples, horizon)
    """
    X, y = [], []
    
    # We must stop early enough so we have enough data for the full horizon target
    for i in range(len(data_array) - seq_length - horizon + 1):
        # X is the past sequence
        X.append(data_array[i : i + seq_length, :]) 
        
        # y is a slice of the exact next 24 hours
        y.append(data_array[i + seq_length : i + seq_length + horizon, target_col_idx])
        
    return np.array(X), np.array(y)

def flatten_for_ml(X_seq):
    n_samples, seq_length, n_features = X_seq.shape
    return X_seq.reshape(n_samples, seq_length * n_features)


def generate_gap_report(df, freq_minutes=15):
    """
    classifies all missing gaps into our 3 Thesis Bins.
    """
    print(f"\nGAP ANALYSIS REPORT ({freq_minutes}-MIN GRID)")
    
    steps_per_hour = int(60 / freq_minutes)
    threshold_small = int(steps_per_hour * config.IMPUTATION_BINS["bin1_threshold_hours"])
    threshold_medium = int(steps_per_hour * config.IMPUTATION_BINS["bin2_threshold_hours"])
    
    for col in df.columns:
        total_nans = df[col].isna().sum()
        if total_nans == 0:
            continue
            
        # Group consecutive NaNs together
        mask = df[col].isna()
        groups = mask.ne(mask.shift()).cumsum()
        gap_sizes = df.groupby(groups)[col].size()[mask.groupby(groups).first()]
        
        # Categorize into Bins
        bin1 = gap_sizes[gap_sizes <= threshold_small]
        bin2 = gap_sizes[(gap_sizes > threshold_small) & (gap_sizes <= threshold_medium)]
        bin3 = gap_sizes[gap_sizes > threshold_medium]
        
        print(f"\n[{col}] - Total Missing Rows: {total_nans}")
        print(f"  🟢 Bin 1 (Safe Interpolation, <= 2h):  {len(bin1)} gaps")
        print(f"  🟡 Bin 2 (Causal Day-Type Blend, <= 1w): {len(bin2)} gaps")
        print(f"  🔴 Bin 3 (Macro-Patch, > 1w):            {len(bin3)} gaps")
        
        # If there are Bin 3 gaps, print their exact dates
        if len(bin3) > 0:
            print("CRITICAL BIN 3 GAPS FOUND:")
            for group_id in bin3.index:
                gap_indices = df[groups == group_id].index
                print(f"From {gap_indices.min()} to {gap_indices.max()} ({len(gap_indices)} steps)")