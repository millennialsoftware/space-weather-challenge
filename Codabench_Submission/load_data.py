"""
Data Loading and Preprocessing for Space Weather Prediction.

This module provides functions for loading, processing, and unifying space weather 
datasets, including OMNI2 and GOES data, for machine learning predictions. It handles 
data cleaning, feature engineering, and time-series transformations required for 
training and inference.

Functions:
    - load_data_for_file_id: Loads and processes space weather data for a given file ID.
    - unify_time_series: Merges OMNI2 and GOES datasets into a common 10-minute interval.
    - create_single_sample_ts: Extracts a time-series sample for machine learning.
    - scale_features: Scales numerical features while preserving binary flags.
"""

# Imports
import pandas as pd
import numpy as np
import os
from features import *
from typing import Tuple, Optional
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
        path (str): Parent directory of dataset
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
    df_omni = pd.read_csv(f"{path}/omni2/omni2-{file_id:05d}.csv",
                          usecols=omni2_fields,
                          parse_dates=['Timestamp'],
                          date_format='%Y-%m-%d %H:%M:%S')

    df_omni = clean_omni2(df_omni)  # Clean OMNI2 dataset

    # Only use last 1440 rows. Some files contain more
    if len(df_omni) > 1440:
        print("\nOMNI2 length > 1440\n")
        df_omni = df_omni.iloc[len(df_omni) - 1440:]

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

    # Earth gravitational parameter
    MU = 398600.4418  # km^3/s^2

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
    M0 = E0 - e * np.sin(E0)  # Initial Eccentric Anomaly
    n  = np.sqrt(MU / a**3)  # Initial Mean Anomaly (rad)

    # Build 10-min time index
    start = T0 - pd.Timedelta(days=59, hours=23, minutes=50)
    end   = T0 + pd.Timedelta(days=2, hours=23, minutes=50)
    idx   = pd.date_range(start=start, end=end, freq='10min')

    # Propagate mean anomaly
    dt = (idx - T0).total_seconds()
    M  = M0 + n * dt

    # Solve Kepler's equation 
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
 
    # drop the constant columns that we propagated 
    propagated_cols = [
        'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)',
        'Altitude (km)'
    ]

    combined.drop(columns=propagated_cols, inplace=True)
   
    # Local Solar Time and cylical transformation.  Note: we calculate solar time again here but with the propagated lon. 
    lon_h = lon / 15.0
    lst = (idx.hour + idx.minute/60 + lon_h) % 24
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

    # Shadow durations
    # flag==1 in eclipse, else 0
    combined['in_eclipse'] = (combined['sunlit_flag']==0).astype(int)

    # Time since eclipse entry (hrs)
    combined['ecl_grp'] = (combined['in_eclipse'] != combined['in_eclipse'].shift()).cumsum()
    combined['time_since_eclipse'] = combined.groupby('ecl_grp').cumcount() * (10/60)
    combined.loc[combined['in_eclipse']==0, 'time_since_eclipse'] = 0

    # Time until next eclipse entry (hrs)
    rev = combined.iloc[::-1].copy()
    rev['time_until_eclipse'] = rev.groupby((rev['in_eclipse'] != rev['in_eclipse'].shift()).cumsum())\
                                .cumcount() * (10/60)
    combined['time_until_eclipse'] = rev['time_until_eclipse'][::-1].values

    # Storm-phase elapsed time
    main = (combined['storm_phase']==2).astype(int)
    combined['main_grp'] = (main != main.shift()).cumsum()
    combined['hours_since_main_start'] = combined.groupby('main_grp').cumcount() * (10/60)
    combined.loc[combined['storm_phase']!=2, 'hours_since_main_start'] = 0

    # Multi-scale rolling stats for key drivers
    wins = [(36,'6h'), (72,'12h'), (144,'24h')]
    for col in ['Bz_neg','SW_Plasma_Speed_km_s','epsilon','Kp_index']:
        for win, label in wins:
            combined[f'{col}_mean_{label}'] = combined[col].rolling(win, min_periods=1).mean()
            combined[f'{col}_std_{label}']  = combined[col].rolling(win, min_periods=1).std().fillna(0)

    # F10.7 lags & rolling trends
    combined['f107_1d_mean'] = combined['f10.7_index'].rolling(144, min_periods=1).mean()
    combined['f107_3d_mean'] = combined['f10.7_index'].rolling(432, min_periods=1).mean()
    combined['f107_lag_1d']  = combined['f10.7_index'].shift(144)
    combined['f107_lag_3d']  = combined['f10.7_index'].shift(432)

    # Third-harmonic DOY/hour
    doy = combined.index.dayofyear.values
    hrs = combined.index.hour.values + combined.index.minute.values/60
    combined['doy_sin3']  = np.sin(6*np.pi * doy/365)
    combined['doy_cos3']  = np.cos(6*np.pi * doy/365)
    combined['hour_sin3'] = np.sin(6*np.pi * hrs/24)
    combined['hour_cos3'] = np.cos(6*np.pi * hrs/24)

    # Nonlinear interaction terms
    combined['E_y_sunlit']    = combined['E_y'] * combined['sunlit_flag']
    combined['Bz_neg_vs2']    = combined['Bz_neg'] * combined['SW_Plasma_Speed_km_s']**2
    combined['pressure_sq']   = combined['Bz_neg_x_pressure']**2

    # Normalized time‐in‐sequence
    n = len(combined)
    combined['time_frac'] = np.linspace(0, 1, n)

    # Clean up temporary grouping cols 
    combined.drop(columns=['ecl_grp','main_grp','in_eclipse'], inplace=True)

    # fill any nans 
    combined.ffill(inplace=True)
    combined.bfill(inplace=True)
     
    # Define max values for GOES features
    goes_max = {
        #"xrsa_flux_mean": 5.74865405e-09,
        "xrsa_flux_max": 7.2619972e-09,
        #"xrsa_flux_min": 4.4141175e-09,
        #"xrsa_flux_std": 9.024464868685411e-10,
        #"xrsb_flux_mean": 1.77766633e-07,
        "xrsb_flux_max": 1.9037122e-07,
        #"xrsb_flux_min": 1.642704e-07,
        #"xrsb_flux_std": 6.201941876503251e-09
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
                'xrsa_flux': ['max'],
                'xrsb_flux': ['max']
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