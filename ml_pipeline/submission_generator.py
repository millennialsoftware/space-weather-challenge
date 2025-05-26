"""
submission_generator.py

This module generates and evaluates satellite density predictions using a trained machine learning model. 
It processes time-series space weather data, applies feature engineering techniques, and saves predictions 
to a JSON file. The generated predictions are then compared to true values to determine model performance.

Main Functions:
    - `generate_submission`: Loads initial state data, prepares time-series inputs, makes predictions, and 
      saves results to a JSON file.
    - `load_data_for_file_id`: Loads and processes space weather data (OMNI2 and optionally GOES) for a given file ID.
    - `unify_time_series`: Merges and resamples time-series data into a unified format with optional lag features.
    - `create_single_sample_ts`: Extracts a single time-series sample for prediction.
    - `scale_features`: Scales numerical features while preserving binary flags.
"""

# Imports
import pandas as pd
import os
import uuid
from datetime import timedelta
import numpy as np
import shutil
import uuid
from model_eval import generate_score
from preprocessing.load_data import load_initial_states
from preprocessing.features import *
import json
import glob
import pickle
import shutil
from typing import Tuple, Optional, Any
from sklearn.base import TransformerMixin

def load_data_for_file_id(
    path: str,
    file_id: int, 
    df_init: pd.DataFrame, 
    include_goes: bool, 
    omni2_fields: list[str]
) -> Tuple[pd.Series, pd.DataFrame, Optional[pd.DataFrame], bool]:
    """Loads and processes space weather data for a given file ID.

    This function retrieves data from multiple files, including OMNI2
    and optionally GOES data. It ensures data quality by applying cleaning functions and handling 
    missing or insufficient GOES data by returning a placeholder DataFrame.

    Args:
        path (str): Path to dataset
        file_id (int): Unique identifier for the data file.
        df_init (pd.DataFrame): DataFrame containing initial states dataset.
        include_goes (bool): Whether to include GOES satellite data.
        omni2_fields (list): List of column names to select from the OMNI2 dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - row (pd.Series): Initial state data for the given `file_id`.
            - df_omni (pd.DataFrame): Processed OMNI2 file.
            - df_goes (pd.DataFrame or None): Processed GOES file (if included) or `None`.
            - goes_is_missing (bool): Flag indicating whether the GOES data was missing or insufficient.
    """
    # Extract the initial state data
    row = df_init[df_init['File ID'] == file_id].iloc[0]

    # Load OMNI2 dataset
    df_omni = pd.read_csv(f"{path}omni2/omni2-{file_id:05d}.csv",
                          usecols=omni2_fields,
                          parse_dates=['Timestamp'],
                          date_format='%Y-%m-%d %H:%M:%S')

    df_omni = clean_omni2(df_omni)  # Clean OMNI2 dataset

    # Only use last 1440 rows. Some files contain more
    if len(df_omni) > 1440:
        df_omni = df_omni.iloc[len(df_omni) - 1440:]

    if len(df_omni) != 1440:
        raise ValueError("Dataset length not 1440 rows")

    # Initialize GOES missing flag
    goes_is_missing = False
    df_goes = None  # Default to None if not using GOES data

    if include_goes:
        EXPECTED_GOES_COLS = ['Timestamp', 'xrsa_flux', 'xrsb_flux']
        goes_file = f"{path}goes/goes-{file_id:05d}.csv"

        if os.path.exists(goes_file) and os.path.getsize(goes_file) > 0:
            try:
                with open(goes_file, 'r') as f:
                    header_line = f.readline().strip()

                if not header_line:
                    raise ValueError("File is empty (no header found).")

                header_cols = [col.strip() for col in header_line.split(',')]

                if not set(EXPECTED_GOES_COLS).issubset(header_cols):
                    raise ValueError(f"Expected columns {EXPECTED_GOES_COLS} not found in file header")

                df_goes = pd.read_csv(goes_file, usecols=EXPECTED_GOES_COLS, parse_dates=['Timestamp'],
                                      date_format='%Y-%m-%d %H:%M:%S', low_memory=False)

                df_goes = clean_goes(df_goes)  # Clean GOES dataset

                # If GOES DataFrame is empty, mark as missing
                if df_goes.empty or df_goes.dropna(subset=["xrsa_flux", "xrsb_flux"]).empty:
                    goes_is_missing = True
                    df_goes = pd.DataFrame(columns=EXPECTED_GOES_COLS)

                if df_goes.shape[0] < 2000:  # If fewer than 2000 rows, treat as missing
                    goes_is_missing = True
                    df_goes = pd.DataFrame(columns=EXPECTED_GOES_COLS)  # Keep column structure

            except (pd.errors.EmptyDataError, ValueError) as e:
                # print(f"Error loading GOES data: {e}")  # Optional debugging log
                goes_is_missing = True
                df_goes = pd.DataFrame(columns=EXPECTED_GOES_COLS)  # Keep column structure

        else:
            goes_is_missing = True
            df_goes = pd.DataFrame(columns=EXPECTED_GOES_COLS)  # Keep column structure

    return row, df_omni, df_goes, goes_is_missing

def unify_time_series(
    row: pd.Series,
    df_omni: pd.DataFrame,
    df_goes: Optional[pd.DataFrame],
    goes_is_missing: bool = False,
    add_lags: bool = False,
    add_flags: bool = False
) -> pd.DataFrame:
    """Unifies time-series data (OMNI2 and GOES) into a common 10-minute interval.

    This function processes and aligns multiple time-series datasets into a unified 10-minute resolution 
    indexed DataFrame. It interpolates missing values, adds orbital metadata, handles missing GOES data, 
    and optionally generates lagged features and missing data flags.

    Args:
        row (pd.Series): A row containing metadata with timestamps and orbital parameters.
        df_omni (pd.DataFrame): The OMNI2 dataset containing solar wind and geomagnetic data.
        df_goes (pd.DataFrame): The GOES dataset containing X-ray flux data.
        goes_is_missing (bool, optional): Whether the GOES dataset is missing entirely. Defaults to False.
        add_lags (bool, optional): Whether to compute and add lagged features. Defaults to False.
        add_flags (bool, optional): Whether to add missing data flags. Defaults to False.

    Returns:
        pd.DataFrame: A unified DataFrame with a 10-minute time index, merged data, and optional features.
    """

    
    T0 = pd.to_datetime(row['Timestamp'])
    start_time = T0 - pd.Timedelta(days=59, hours=23, minutes=50)
    end_time = T0 + pd.Timedelta(days=2, hours=23, minutes=50)

    # Generate a 10-minute DateTime index
    idx_10min = pd.date_range(start=start_time, end=end_time, freq='10min')

    # Resample OMNI2 dataset
    df_omni = df_omni.set_index('Timestamp').sort_index()
    df_omni = df_omni.loc[(df_omni.index >= start_time) & (df_omni.index <= T0)]
    df_omni_10 = df_omni.reindex(idx_10min, method='nearest').interpolate(method='linear')

    # Resample satellite density dataset
    df_density = df_density.set_index('Timestamp').sort_index()
    df_density_10 = df_density.reindex(idx_10min, method='nearest')

    # Combine datasets
    combined = pd.DataFrame(index=idx_10min)
    combined = pd.concat([combined, df_omni_10], axis=1)
    combined['log_density'] = df_density_10['log_density']

    # Add orbital metadata as constants, excluding Timestamp and File ID
    exclude_cols = ["Timestamp", "File ID"]
    for col in row.index: 
        if col not in exclude_cols:
            combined[col] = row[col]

    ## Propagation 
    # Earth gravitational parameter
    MU = 398600.4418  # km^3/s^2

    #  Get constant orbital elements from inital state 
    a = row['Semi-major Axis (km)']
    e = row['Eccentricity']
    incl  = np.deg2rad(row['Inclination (deg)'])
    raan  = np.deg2rad(row['RAAN (deg)'])
    argp  = np.deg2rad(row['Argument of Perigee (deg)'])
    nu0   = np.deg2rad(row['True Anomaly (deg)'])

    # Compute initial anomalies for propagation
    E0 = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(nu0/2))
    M0 = E0 - e * np.sin(E0)  # Initial Eccentric Anomaly
    n  = np.sqrt(MU / a**3)  # Initial Mean Anomaly (rad)

    # Propagate mean anomaly
    dt = (idx_10min - T0).total_seconds()
    M  = M0 + n * dt

    # Solve Kepler's equation 
    def solve_kepler(M, e, iters=5):
        E = M.copy()
        for _ in range(iters):
            E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        return E
    
    E = solve_kepler(M, e)

    # True anomaly and orbital radius
    nu = 2 * np.arctan2(np.sqrt(1 + e)*np.sin(E/2), np.sqrt(1 - e)*np.cos(E/2))
    r  = a * (1 - e*np.cos(E))

    # Perifocal frame
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    
    # Rotate to ECI coordinates
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_w, sin_w = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(incl), np.sin(incl)
    x_eci = (cos_O*cos_w - sin_O*sin_w*cos_i)*x_p + (-cos_O*sin_w - sin_O*cos_w*cos_i)*y_p
    y_eci = (sin_O*cos_w + cos_O*sin_w*cos_i)*x_p + (-sin_O*sin_w + cos_O*cos_w*cos_i)*y_p
    z_eci = (sin_w*sin_i)*x_p + (cos_w*sin_i)*y_p
    
    # Calculate propagated lat/lon/alt
    lat = np.degrees(np.arctan2(z_eci, np.sqrt(x_eci**2 + y_eci**2)))
    lon = np.degrees(np.arctan2(y_eci, x_eci))
    
    combined['Latitude_dyn']  = lat
    combined['Longitude_dyn'] = lon
    combined['altitude_dyn']  = r - 6371.0

    # Cyclical transforms of lat/long and true anamoly 
    combined['anom_sin'] = np.sin(nu)
    combined['anom_cos'] = np.cos(nu)
    lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
    combined['lat_sin'] = np.sin(lat_rad)
    combined['lat_cos'] = np.cos(lat_rad)
    combined['lon_sin'] = np.sin(lon_rad)
    combined['lon_cos'] = np.cos(lon_rad)
 
    # Drop the constant columns that we propagated 
    propagated_cols = [
        'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)',
        'Altitude (km)'
    ]

    combined.drop(columns=propagated_cols, inplace=True)
   
    # Local Solar Time and cylical transformation. 
    lon_h = lon / 15.0
    lst = (idx_10min.hour + idx_10min.minute/60 + lon_h) % 24
    combined['lst_sin'] = np.sin(2*np.pi*lst/24)
    combined['lst_cos'] = np.cos(2*np.pi*lst/24)

    # Sunlit Flag/solar_zenith. compute_solar_zenith helper is in features.py.
    combined['solar_zenith'] = compute_solar_zenith(
        times=idx, lat=combined['Latitude_dyn'], lon=combined['Longitude_dyn']
    )

    # Flag to tell if satellite is in day light or not using solar_zeinth angle 
    combined['sunlit_flag'] = (combined['solar_zenith'] < 90).astype(int)

    # Storm Phase Indicators. Kp-Index is between 0-90. Use that to set threshholds to indicate 3 conditions. 
    combined['kp_diff_1h'] = combined['Kp_index'] - combined['Kp_index'].shift(6)
    conds = [
        (combined['Kp_index'] < 50) & (combined['kp_diff_1h'] > 0), # pre-storm
        (combined['Kp_index'] >= 50),                               # main phase
        (combined['kp_diff_1h'] < 0)                                # recovery
    ]

    choices = [1, 2, 3]
    combined['storm_phase'] = np.select(conds, choices, default=0)

    # Negative Bz Component. Measures distinct influences of southward vs. northward interplanetary magnetic fields. 
    combined['Bz_neg'] = (-combined['BZ_nT_GSM']).clip(lower=0)
    combined['Bz_pos'] = combined['BZ_nT_GSM'].clip(lower=0)

    # This measures a strong energy input into the upper atmosphere. When Bz_neg (southward)
    combined['Bz_neg_x_pressure'] = combined['Bz_neg'] * combined['Flow_pressure']

    # Fast solar wind speed tend to drive stronger geomagnetic activity (Kp_index). This attempts to measures when both are elevated. 
    combined['speed_x_kp'] = combined['SW_Plasma_Speed_km_s'] * combined['Kp_index']

    # This attempts to give the direction the magnetic field is pointing to the model. 
    clock = np.arctan2(combined['BY_nT_GSM'], combined['BZ_nT_GSM'])
    combined['clock_sin'] = np.sin(clock)
    combined['clock_cos'] = np.cos(clock)
    
    # The overall strength of the interplanetary magnetic field,
    combined['B_mag'] = np.sqrt(
        combined['BX_nT_GSE_GSM']**2 +
        combined['BY_nT_GSM']**2    +
        combined['BZ_nT_GSM']**2
    )

    # Measures how solar wind’s motion and magnetic field work together to drive energy into Earth’s upper atmosphere
    combined['E_y'] = -combined['SW_Plasma_Speed_km_s'] * combined['BZ_nT_GSM']
    
    # Estimates the rate at which the solar wind transfers energy into the magnetosphere.
    combined['epsilon'] = (
        combined['SW_Plasma_Speed_km_s'] *
        combined['B_mag']**2 *
        np.sin(clock/2)**4
    )

    # sin/cos transformation of DOY and HOUR
    doy = combined.index.dayofyear.values
    hrs = combined.index.hour.values + combined.index.minute.values/60
    combined['doy_sin2'] = np.sin(4 * np.pi * doy / 365)
    combined['doy_cos2'] = np.cos(4 * np.pi * doy / 365)
    combined['hour_sin2'] = np.sin(4 * np.pi * hrs / 24)
    combined['hour_cos2'] = np.cos(4 * np.pi * hrs / 24)
    
    # Drop raw DOY/HOUR
    time_feats =[ 'DOY', 'Hour']
    combined.drop(columns=time_feats, inplace=True)

    # fill any nans 
    combined.ffill(inplace=True)
    combined.bfill(inplace=True)
     
    # Define max values for GOES features
    goes_max = {
        "xrsa_flux_mean": 5.74865405e-09,
        "xrsa_flux_max": 7.2619972e-09,
        "xrsa_flux_min": 4.4141175e-09,
        "xrsa_flux_std": 9.024464868685411e-10,
        "xrsb_flux_mean": 1.77766633e-07,
        "xrsb_flux_max": 1.9037122e-07,
        "xrsb_flux_min": 1.642704e-07,
        "xrsb_flux_std": 6.201941876503251e-09
    }

    # Process GOES data
    should_flag_goes = False
    if df_goes is not None:

        if goes_is_missing or df_goes is None or df_goes.empty:

            # Fill all GOES columns with max values if data is missing
            for col, median_val in goes_max.items():
                combined[col] = median_val
            should_flag_goes = True
        else:
            # Set the Timestamp as index, sort, and filter for the desired time window
            df_goes = df_goes.set_index('Timestamp').sort_index()
            df_goes = df_goes.loc[(df_goes.index >= start_time) & (df_goes.index <= T0)]

            # Process real GOES data using the already set index and filtered df_goes
            aggregation_dict = {
                'xrsa_flux': ['max','mean','min','std'],
                'xrsb_flux': ['max','mean','min','std']
            }
            df_goes_10 = df_goes.resample('10min').agg(aggregation_dict)
            df_goes_10.columns = [f"{col[0]}_{col[1]}" for col in df_goes_10.columns]
            
            if add_flags:
                df_goes_10["Goes_Missing_Flag"] = df_goes_10.isna().any(axis=1).astype(int)

            df_goes_10 = df_goes_10.reindex(idx_10min, method='nearest').interpolate()
            for col in df_goes_10.columns:
                combined[col] = df_goes_10[col]

    if add_lags:
        # Define lag groups
        lag_dict = {
            'SW_Proton_Density_N_cm3': [6, 12, 18],
            'SW_Plasma_Speed_km_s': [6, 12, 18],
            'Flow_pressure': [6, 12, 18],
            'Kp_index': [12, 18, 36, 48],
            'ap_index_nT': [12, 18, 36, 48],
            'Dst_index_nT': [12, 18, 36, 48],
            'BZ_nT_GSM': [6, 12, 36],
            'Magnetosonic_Mach_number': [6, 12, 36],
            'Plasma_Beta': [6, 12, 36]
        }

        for col, lags in lag_dict.items():
            if col in combined.columns:
                for lag in lags:
                    combined[f"{col}_lag_{lag}"] = combined[col].shift(lag)

        if df_goes is not None:
            xray_lags = [6, 12]
            xray_columns = [
                'xrsa_flux_mean', 'xrsa_flux_max', 'xrsa_flux_min', 'xrsa_flux_std',
                'xrsb_flux_mean', 'xrsb_flux_max', 'xrsb_flux_min', 'xrsb_flux_std'
            ]
            for col in xray_columns:
                if col in combined.columns:
                    for lag in xray_lags:
                        combined[f"{col}_lag_{lag}"] = combined[col].shift(lag)

        combined.ffill(inplace=True)
        combined.bfill(inplace=True)

    if add_flags:
        # Define medians for missing value imputation
        omni_medians = {
            "YEAR": 2010.0,
            "DOY": 186.0,
            "Hour": 11.0,
            "Bartels_rotation_number": 2408.0,
            "num_points_IMF_averages": 59.0,
            "num_points_Plasma_averages": 35.0,
            "Lat_Angle_of_B_GSE": -0.1,
            "Long_Angle_of_B_GSE": 183.2,
            "BX_nT_GSE_GSM": 0.0,
            "BY_nT_GSE": -0.1,
            "BZ_nT_GSE": 0.0,
            "BY_nT_GSM": -0.1,
            "BZ_nT_GSM": 0.0,
            "RMS_magnitude_nT": 0.2,
            "RMS_field_vector_nT": 1.7,
            "RMS_BX_GSE_nT": 0.8,
            "RMS_BY_GSE_nT": 0.9,
            "RMS_BZ_GSE_nT": 1.0,
            "SW_Plasma_Temperature_K": 65110.0,
            "SW_Proton_Density_N_cm3": 4.7,
            "SW_Plasma_Speed_km_s": 404.0,
            "SW_Plasma_flow_long_angle": -0.2,
            "SW_Plasma_flow_lat_angle": -0.8,
            "sigma_T_K": 8889.0,
            "sigma_n_N_cm3": 0.4,
            "sigma_V_km_s": 5.0,
            "sigma_phi_V_degrees": 0.8,
            "sigma_theta_V_degrees": 0.8,
            "Kp_index": 13.0,
            "R_Sunspot_No": 26.0,
            "Dst_index_nT": -9.0,
            "ap_index_nT": 5.0,
            "f10.7_index": 80.9,
            "AE_index_nT": 91.0,
            "AL_index_nT": -46.0,
            "AU_index_nT": 40.0,
            "pc_index": 0.6
        }

        # Define columns to flag
        flag_columns = []

        existing_columns = [col for col in omni_medians.keys() if col in combined.columns]
        if existing_columns:
            combined["Omni_Missing_Flag"] = combined[existing_columns].isna().any(axis=1).astype(int)
            flag_columns.append('Omni_Missing_Flag')

            for col in existing_columns:
                combined[col] = combined[col].fillna(omni_medians[col])
            
        if df_goes is not None and 'Goes_Missing_Flag' not in combined.columns:
            combined['Goes_Missing_Flag'] = int(should_flag_goes)
            flag_columns.append('Goes_Missing_Flag')

        # Reorder columns to place flags at the end
        other_columns = [col for col in combined.columns if col not in flag_columns]
        combined = combined[other_columns + flag_columns]
   
        # Set any nan values to 0
        if combined.isna().sum().sum() > 0:
            print(combined.isna().sum())
            raise ValueError("NaNs still in dataset")
        
    # Remove any remaining nans - should be none but needed nonetheless.
    combined = combined.fillna(0)

    return combined

def create_single_sample_ts(
    df: pd.DataFrame, 
    T0: pd.Timestamp, 
    lookback: int = (30 * 24 * 6), 
) -> np.ndarray:
    """Extracts a single time-series sample from the dataset for model training.

    This function slices the dataset into past samples based on a given timestamp (`T0`).
    It returns a `lookback` period of historical data (`X`).

    Args:
        df (pd.DataFrame): The input time-series DataFrame containing space weather and satellite data.
        T0 (pd.Timestamp): The timestamp that defines the split between past and future data.
        lookback (int, optional): The number of past time steps to include (default: 30 days * 24 hours * 6 steps/hour).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): The past data with shape `(lookback, num_features)`

    Raises:
        ValueError: If `T0` is not found in the DataFrame index or if there is insufficient historical data.
    """
    try:
        end_of_history = df.index.get_loc(T0)  # Get integer location of T0
    except KeyError:
        raise ValueError(f"T0 timestamp {T0} not found in DataFrame index.")

    if isinstance(end_of_history, slice):  # Handle slice cases
        end_of_history = end_of_history.stop

    start_of_history = end_of_history - lookback

    # Ensure sufficient historical data
    if start_of_history < 0:
        raise ValueError("Not enough data for the 60-day window.")

    df_past = df.iloc[start_of_history:end_of_history]

    # Extract past features
    X = df_past.to_numpy()

    return X

def scale_features(
    scaler: TransformerMixin,
    X: np.ndarray,
    start_flag_index: Optional[int] = None
):
    """Scales features while keeping binary flags unchanged.

    This function scales only the numerical features in 2D time-series datasets, 
    preserving any binary flags present in the latter part of the feature set.

    Args:
        scaler (TransformerMixin): A scikit-learn scaler instance (e.g., MinMaxScaler, StandardScaler).
        X (np.ndarray): Training dataset with shape (num_samples, seq_len, num_features).
        start_flag_index (Optional[int], optional): Index where binary flags begin. 
            - Features before this index are scaled.
            - If None, all features are scaled. Defaults to None.

    Returns:
        np.ndarray: 
            - Scaled and reshaped data (`X_input`).
    """
    if start_flag_index:
        X_features = X[:, :start_flag_index]
        X_flags = X[:, start_flag_index:]

        X_features_scaled = scaler.transform(X_features)
        X_scaled = np.concatenate([X_features_scaled, X_flags], axis=1)
    else:
        X_scaled = scaler.transform(X)

    X_input = X_scaled[np.newaxis, :, :]

    return X_input

def generate_submission(
    selection: str,
    path: str,
    model: Any,
    X_scaler: TransformerMixin,
    y_scaler: TransformerMixin,
    include_goes: bool,
    add_lags: bool,
    add_flags: bool,
    omni2_fields: list[str],
    best_submission_path: Optional[str],
    show_status: bool = False,
    transformer: str = None,
) -> Optional[tuple[float, dict]]:
    """Generates and saves satellite density predictions using a trained model.

    This function loads initial state data, processes time series inputs, makes 
    predictions using the provided model, and saves the results to a JSON file. 
    It also evaluates model performance and allows the user to save the model if 
    they desire. 
    
    Args:
        selection (str): String name of trained model
        path (str): Path to dataset.
        model (Any): A trained machine learning model used for prediction.
        X_scaler (TransformerMixin): A fitted scaler for feature normalization.
        y_scaler (TransformerMixin): A fitted scaler for target variable normalization.
        include_goes (bool): Whether to include GOES satellite data in the input.
        add_lags (bool): Whether to include lagged features in the input data.
        add_flags (bool): Whether to include binary flag features in the input data.
        omni2_fields (list[str]): A list of Omni2 dataset fields to include.
        best_submission_path (Optional[str], optional): 
            - If `None`, the function compares model performance interactively and may save the best model.
            - If a valid file path is provided, the function compares the new predictions to this file and 
              returns the comparison score along with the predictions.
        show_status (bool, optional): If True, prints status updates. Defaults to False.
        transformer (str, optional): If string is passed in, custo transformer block is fetched
                                     and saved to submission fil.
    Returns:
        - If `best_submission_path` is `None`: Returns `None`. The function saves predictions to a JSON file 
          and may update the best model based on performance.
        - If `best_submission_path` is provided: Returns a tuple `(ps, predictions)`, where:
            - `ps` (float): The propagation score comparing the new submission to the reference submission.
            - `predictions` (dict): A dictionary containing timestamped density predictions.
    """
    # Set file name
    unique_id = uuid.uuid4()
    file_name = f"submissions/submission_{unique_id}.json"

    if show_status:
        print("Loading Data") # Start of data loading

    # Load in initial states
    _, df_init = load_initial_states(path=path)

    # Convert 'Timestamp' to datetime
    df_init['Timestamp'] = pd.to_datetime(df_init['Timestamp'])

    # Create empty dictionary for predictions
    predictions = {}

    # Iterate through each row, generating predictions for each file id.
    for _, row in df_init.iterrows():
        
            # Get prediction interval
            initial_time = row['Timestamp'].ceil('10min')
            end_time = initial_time + timedelta(days=2, hours=23, minutes=50)

            # Generate a range of timestamps every 10 minutes for the prediction interval
            timestamps = pd.date_range(start=initial_time, end=end_time, freq='10min')

            # Create a predictions DataFrame with the generated timestamps
            predictions_df = pd.DataFrame({'Timestamp': timestamps})    

            # Set the file id
            file_id = int(row['File ID'])
        
            if show_status:
                print(f"Predicting for {file_id}") # Print start of prediction for file

            # Load in data for a file id
            row, df_omni, df_goes, goes_is_missing = load_data_for_file_id(
                path, file_id, df_init, include_goes, omni2_fields=omni2_fields
            )

            # Unify time series together
            df_10min = unify_time_series(
                row, df_omni, df_goes, goes_is_missing, 
                add_lags=add_lags, add_flags=add_flags
            )

            # Create a single sample of data using a sliding window technique
            X = create_single_sample_ts(df_10min, T0=row['Timestamp'])

            # Set binary flags to be ignored from scaling 
            num_feats = X.shape[1]  # Total number of features

            # Number of flags
            num_flags = 0

            # Check if add_flags is True and at least one feature in feature_array is in omni2_fields
            if add_flags:
                num_flags += 1
                if include_goes:
                    num_flags += 1

            # Set start_flag_index only if previous conditions are met
            if add_flags and num_flags > 0:
                start_flag_index = num_feats - num_flags
            else:
                start_flag_index = None

            # Scale data 
            X_input = scale_features(X_scaler, X, start_flag_index)

            # Predict density values 
            y_pred_scaled = model.predict(X_input, verbose=0)

            if np.isnan(y_pred_scaled).sum() > 0:
                print(f"NaN predictions for file id {file_id}") # Update if predictions are nan
                raise ValueError("NaNs in output!")

            # Undo Min-Max Scaling for predicted values
            y_pred_scaled_flat = y_pred_scaled.reshape(-1, 1)
            y_pred_flat = y_scaler.inverse_transform(y_pred_scaled_flat)

            # Undo the log transformation for predicted values
            y_pred = np.exp(y_pred_flat)

            # Add predictions to prediction dataframe
            predictions_df['Orbit Mean Density (kg/m^3)'] = y_pred.flatten()
            
            # Load true values (satellite density)
            sat_density_file_pattern = os.path.join(f"{path}sat_density/", f"*-{file_id:05d}.csv")
            matching_files = glob.glob(sat_density_file_pattern)
            sat_density_file = matching_files[0] if matching_files else ""
            df_density = pd.read_csv(sat_density_file, parse_dates=['Timestamp'], date_format='%Y-%m-%d %H:%M:%S')

            # Clean the density data
            df_density = clean_sat_density(df_density)  

            # Ensure timestamps are aligned
            df_density = df_density.set_index("Timestamp").reindex(predictions_df["Timestamp"], method='nearest')

            # Check for remaining NaNs
            nan_count = df_density["Orbit Mean Density (kg/m^3)"].isna().sum()
            if nan_count > 0:
                continue

            # Add cleaned density data to predictions
            predictions_df["True Orbit Mean Density (kg/m^3)"] = df_density["Orbit Mean Density (kg/m^3)"].values

            # Add predictions to dictionary 
            predictions[file_id] = {
                    "Timestamp": list(map(lambda ts: ts.isoformat(), predictions_df["Timestamp"])),
                    "Pred Orbit Mean Density (kg/m^3)": predictions_df['Orbit Mean Density (kg/m^3)'].tolist(),
                    "Orbit Mean Density (kg/m^3)": predictions_df['True Orbit Mean Density (kg/m^3)'].tolist(),
            }

            if show_status:
                print(f"Model execution for {file_id} Finished")  # Print that model finished for a file

    # Save the predictions to a JSON file
    with open(file_name, "w") as outfile:
            json.dump(predictions, outfile)
        
    if show_status:
        print(f"Saved predictions to: {file_name}")  # Confirm predictions saved

    # Check if submission path was passed in
    if not best_submission_path:

        # If path does not exist save this model as best model.
        if os.path.exists("submissions/best_submission.json"):

            # Get propagation score
            ps = generate_score(file_name)

            # Check if model performed worse or better
            if ps > 0:
                print("\033[34mModel has higher propagation score\033[0m")
            else:
                print("\033[31mModel has lower propagation score\n\033[0m")

            # See whether or not to save as the best model
            save_best = "y"
            while save_best not in ["y", "n"]:
                save_best = input("\nWould you like to save to Codabench folder (y/n)? ")

            # Decided to save as best model
            if save_best == "y":
                
                # Saved predictions to best submission file
                with open("submissions/best_submission.json", "w") as outfile:
                    json.dump(predictions, outfile)
                    
                print("\033[34mSaving as new best model\033[0m")

                directory = "../Codabench_Submission" 

                # Delete all previous model files
                files_to_delete = glob.glob(os.path.join(directory, "*.pkl")) + glob.glob(os.path.join(directory, "*.keras"))
                for file in files_to_delete:
                    os.remove(file)
                
                if os.path.exists("../Codabench_Submission/transformer.py"):
                    os.remove("../Codabench_Submission/transformer.py")

                # Save new best model files
                model.save(f"{directory}/model.keras")
                with open(f"{directory}/X_scaler.pkl", "wb") as x_scaler_file:
                    pickle.dump(X_scaler, x_scaler_file)

                with open(f"{directory}/y_scaler.pkl", "wb") as y_scaler_file:
                    pickle.dump(y_scaler, y_scaler_file)

                # Save parameters to a JSON file
                parameters = {
                    "add_flags": add_flags,
                    "add_lags": add_lags,
                    "include_goes": include_goes,
                    "omni2_fields": omni2_fields,
                    "transformer": transformer
                }

                # Save to codabench folder
                with open(f"{directory}/parameters.json", "w") as param_file:
                    json.dump(parameters, param_file, indent=4)

                # If transformer model, save model to models folder
                if transformer is not None:
                    destination_path = os.path.join(directory, "transformer.py")
                    source_path = "models/"+transformer +".py"
                    shutil.copy2(source_path, destination_path)

                # Save into the models folder as well
                saved_directory = os.path.join("models", "saved", f"{selection}_{unique_id}")

                shutil.copytree(src=directory, dst=saved_directory)

                print("Model successfully saved!")
            else:
                # See if we should still save model in the models folder
                save = ""
                while save not in ["y", "n"]:
                    save = input("\n\nWould you like to save to saved models folder (y/n)? ")

                # We decided to save in the models folder
                if save == "y":

                    coda_bench_dir = "../Codabench_Submission"
                    saved_directory = os.path.join("models", "saved", f"{selection}_{unique_id}")
                    os.makedirs(saved_directory, exist_ok=True)
                    files_to_copy = ['environment.yml', 'load_data.py', 'submission.py', 'features.py']

                    for file in files_to_copy:
                        source_path = os.path.join(coda_bench_dir, file)
                        destination_path = os.path.join(saved_directory, file)

                        if os.path.isfile(source_path):
                            shutil.copy2(source_path, destination_path)
                            print(f"Copied {file} to {saved_directory}")

                    # Save parameters to a JSON file
                    parameters = {
                        "add_flags": add_flags,
                        "add_lags": add_lags,
                        "include_goes": include_goes,
                        "omni2_fields": omni2_fields,
                        "transformer": transformer
                    }

                    with open(os.path.join(saved_directory, "parameters.json"), "w") as param_file:
                        json.dump(parameters, param_file, indent=4)
                                        
                    # If transformer model, save model to models folder
                    if transformer is not None:
                        destination_path = os.path.join(saved_directory, "transformer.py")
                        source_path = "models/"+transformer +".py"
                        shutil.copy2(source_path, destination_path)

                    # Save model files
                    model.save(os.path.join(saved_directory,"model.keras"))
                    with open(os.path.join(saved_directory, "X_scaler.pkl"), "wb") as x_scaler_file:
                        pickle.dump(X_scaler, x_scaler_file)
                    with open(os.path.join(saved_directory, "y_scaler.pkl"), "wb") as y_scaler_file:
                        pickle.dump(y_scaler, y_scaler_file)

                    print("Model successfully saved!")
        
        else:
            # First one trained so save this one as the best one
            with open("submissions/best_submission.json", "w") as outfile:
                json.dump(predictions, outfile)
    else:
        if os.path.exists("submissions/best_features.json"):
            # Generate propagation score
            ps = generate_score(file_name, best_submission_path)

            # Return both the ps and predictions dictionary for custom logic implementation
            return ps, predictions
        else:
            # First one trained so save this one as the best one
            with open("submissions/best_features.json", "w") as outfile:
                json.dump(predictions, outfile)
            return 0, predictions
