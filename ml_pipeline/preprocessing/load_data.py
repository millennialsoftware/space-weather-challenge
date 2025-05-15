"""
load_data.py

This module provides functions for loading, processing, and preparing space weather datasets
for neural network models. The dataset includes data from OMNI2 (solar wind and geomagnetic
indices), satellite density, and optionally GOES (X-ray flux) files. The module supports time-series
alignment, feature engineering (lags, flags), and conversion into a structured format for model
training.

Key Features:
- Loads and preprocesses multiple space weather data files.
- Handles missing values and provides imputation strategies.
- Aligns data from different sources to a unified 10-minute resolution.
- Generates lagged features and missing data flags (optional).
- Returns feature matrices (`X`) and target variables (`y`) for model input.

Functions:
    - `load_data_for_file_id(path, file_id, df_init, include_goes, omni2_fields)`
        Loads space weather data for a given file ID.
    
    - `unify_time_series(row, df_omni, df_density, df_goes, goes_is_missing, add_lags, add_flags)`
        Unifies time-series data into a common 10-minute interval.

    - `create_single_sample_ts(df, T0, lookback, horizon)`
        Extracts a single time-series sample for model training.

    - `load_initial_states(path, full)`
        Loads the initial states dataset, used to link to OMNI2 files.

    - `load_data(path, num_files, full, val, include_goes, add_lags, omni2_fields, add_flags)`
        Loads and processes multiple space weather data files, returning structured input for models.

Dependencies:
    - pandas
    - numpy
    - tqdm
    - glob
    - os
    - typing (List, Tuple, Optional)
    - Custom feature processing functions (`features` module)
"""

# Imports
import pandas as pd
import numpy as np
import glob
import os
from .features import *
from tqdm import tqdm
from typing import Tuple, Optional, List


def load_data_for_file_id(
    path: str,
    file_id: int, 
    df_init: pd.DataFrame, 
    include_goes: bool, 
    omni2_fields: list[str]
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], bool]:
    """Loads and processes space weather data for a given file ID.

    This function retrieves data from multiple files, including OMNI2, satellite density, 
    and optionally GOES data. It ensures data quality by applying cleaning functions and handling 
    missing or insufficient GOES data by returning a placeholder DataFrame.

    Args:
        path (str): File path for dataset
        file_id (int): Unique identifier for the data file.
        df_init (pd.DataFrame): DataFrame containing initial states dataset.
        include_goes (bool): Whether to include GOES satellite data.
        omni2_fields (list): List of column names to select from the OMNI2 dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - row (pd.Series): Initial state data for the given `file_id`.
            - df_omni (pd.DataFrame): Processed OMNI2 file.
            - df_density (pd.DataFrame): Processed satellite density file with a log-transformed density column.
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

    # Clean OMNI2 dataset
    df_omni = clean_omni2(df_omni)

    # Only use last 1440 rows. Some files contain more
    if len(df_omni) > 1440:
        df_omni = df_omni.iloc[len(df_omni) - 1440:]
   
    # Make sure omni length is 1440
    if len(df_omni) != 1440:
        raise ValueError("Dataset length not 1440 rows")

    # Load satellite density dataset
    sat_density_file_pattern = os.path.join(f"{path}sat_density/", f"*-{file_id:05d}.csv")
    matching_files = glob.glob(sat_density_file_pattern)
    sat_density_file = matching_files[0] if matching_files else ""
    df_density = pd.read_csv(sat_density_file, parse_dates=['Timestamp'], date_format='%Y-%m-%d %H:%M:%S')
    
    # Clean density dataset
    df_density = clean_sat_density(df_density)

    # Add log density for better distribution
    df_density['log_density'] = np.log(df_density['Orbit Mean Density (kg/m^3)'])

    # Initialize GOES missing flag
    goes_is_missing = False
    df_goes = None  # Defaults to None if not using GOES data

    if include_goes:
        # Load in goes data
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

    return row, df_omni, df_density, df_goes, goes_is_missing

def unify_time_series(
    row: pd.Series,
    df_omni: pd.DataFrame,
    df_density: pd.DataFrame,
    df_goes: Optional[pd.DataFrame],
    goes_is_missing: bool = False,
    add_lags: bool = False,
    add_flags: bool = False
) -> pd.DataFrame:
    """Unifies time-series data (OMNI2, satellite density, GOES) into a common 10-minute interval.

    This function processes and aligns multiple time-series datasets into a unified 10-minute resolution 
    indexed DataFrame. It interpolates missing values, adds orbital metadata, handles missing GOES data, 
    and optionally generates lagged features and missing data flags.

    Args:
        row (pd.Series): A row containing orbital parameters.
        df_omni (pd.DataFrame): The OMNI2 dataset containing solar wind and geomagnetic data.
        df_density (pd.DataFrame): The satellite density dataset with log density values.
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

    # Get constant orbital elements from inital state 
    a = row['Semi-major Axis (km)']
    e = row['Eccentricity']
    incl = np.deg2rad(row['Inclination (deg)'])  # convert to radians 
    raan = np.deg2rad(row['RAAN (deg)'])
    argp = np.deg2rad(row['Argument of Perigee (deg)'])
    nu0 = np.deg2rad(row['True Anomaly (deg)'])

    # Compute initial anomalies for propagation
    E0 = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(nu0/2))
    M0 = E0 - e * np.sin(E0)  # Initial Eccentric Anomaly
    n  = np.sqrt(MU / a**3)  # Initial Mean Anomaly (rad)

    # Build 10-min time index
    start = T0 - pd.Timedelta(days=59, hours=23, minutes=50)
    end = T0 + pd.Timedelta(days=2, hours=23, minutes=50)
    idx = pd.date_range(start=start, end=end, freq='10min')

    # Propagate mean anomaly
    dt = (idx - T0).total_seconds()
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
    
    # These could maybe be removed as we have the cylical transformations below 
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
   
    # Local Solar Time and cylical transformation.  
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
        (combined['Kp_index'] < 50) & (combined['kp_diff_1h'] > 0),  # pre-storm
        (combined['Kp_index'] >= 50),                                # main phase
        (combined['kp_diff_1h'] < 0)                               # recovery
    ]

    choices = [1, 2, 3]
    combined['storm_phase'] = np.select(conds, choices, default=0)

    # Negative Bz Component. Measures distinct influences of southward vs. northward interplanetary magnetic fields. 
    combined['Bz_neg'] = (-combined['BZ_nT_GSM']).clip(lower=0)
    combined['Bz_pos'] = combined['BZ_nT_GSM'].clip(lower=0)

    # this measures a strong energy input into the upper atmosphere. When Bz_neg (southward)
    combined['Bz_neg_x_pressure'] = combined['Bz_neg'] * combined['Flow_pressure']

    # Fast solar wind speed tend to drive stronger geomagnetic activity (Kp_index). This attempts to measures when both are elevated. 
    combined['speed_x_kp'] = combined['SW_Plasma_Speed_km_s'] * combined['Kp_index']

    
    #  This attempts to give the direction the magnetic field is pointing to the model. 
    clock = np.arctan2(combined['BY_nT_GSM'], combined['BZ_nT_GSM'])
    combined['clock_sin'] = np.sin(clock)
    combined['clock_cos'] = np.cos(clock)
    
    # The overall strength of the interplanetary magnetic field,
    combined['B_mag'] = np.sqrt(
        combined['BX_nT_GSE_GSM']**2 +
        combined['BY_nT_GSM']**2    +
        combined['BZ_nT_GSM']**2
    )
    # measures how solar wind’s motion and magnetic field work together to drive energy into Earth’s upper atmosphere
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

    #  Shadow durations
    # flag==1 in eclipse, else 0
    combined['in_eclipse'] = (combined['sunlit_flag']==0).astype(int)

    # time since eclipse entry (hrs)
    combined['ecl_grp'] = (combined['in_eclipse'] != combined['in_eclipse'].shift()).cumsum()
    combined['time_since_eclipse'] = combined.groupby('ecl_grp').cumcount() * (10/60)
    combined.loc[combined['in_eclipse']==0, 'time_since_eclipse'] = 0

    # time until next eclipse entry (hrs)
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

    #  Third-harmonic DOY/hour
    doy = combined.index.dayofyear.values
    hrs = combined.index.hour.values + combined.index.minute.values/60
    combined['doy_sin3']  = np.sin(6*np.pi * doy/365)
    combined['doy_cos3']  = np.cos(6*np.pi * doy/365)
    combined['hour_sin3'] = np.sin(6*np.pi * hrs/24)
    combined['hour_cos3'] = np.cos(6*np.pi * hrs/24)

    #  Nonlinear interaction terms
    combined['E_y_sunlit']    = combined['E_y'] * combined['sunlit_flag']
    combined['Bz_neg_vs2']    = combined['Bz_neg'] * combined['SW_Plasma_Speed_km_s']**2
    combined['pressure_sq']   = combined['Bz_neg_x_pressure']**2

    #  Normalized time‐in‐sequence
    n = len(combined)
    combined['time_frac'] = np.linspace(0, 1, n)

    # clean up temporary grouping cols 
    combined.drop(columns=['ecl_grp','main_grp','in_eclipse'], inplace=True)


    # Fill any nans 
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

            df_goes = df_goes.set_index('Timestamp').sort_index()
            df_goes = df_goes.loc[(df_goes.index >= start_time) & (df_goes.index <= T0)]

           
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
                'xrsa_flux_max',
                'xrsb_flux_max',
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

        # Exclude density from nan check - checked later and file is skipped if there are nans
        excluded_column = "log_density" 
        columns_to_check = combined.drop(columns=[excluded_column], errors="ignore")

        # Abort training if NaN values are still present in features
        if columns_to_check.isna().sum().sum() > 0:  # Only adding flags there should be 0 nans
                print(columns_to_check.isna().sum()) 
                raise ValueError("NaN values in combined (excluding {}), aborting training!".format(excluded_column))
        
    return combined

def create_single_sample_ts(
    df: pd.DataFrame, 
    T0: pd.Timestamp, 
    lookback: int = (30 * 24 * 6), 
    horizon: int = 432
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts a single time-series sample from the dataset for model training.

    This function slices the dataset into past and future samples based on a given timestamp (`T0`).
    It returns a `lookback` period of historical data (`X`) and a `horizon` period of future values (`y`).

    Args:
        df (pd.DataFrame): The input time-series DataFrame containing space weather and satellite data.
        T0 (pd.Timestamp): The timestamp that defines the split between past and future data.
        lookback (int, optional): The number of past time steps to include (default: 30 days * 24 hours * 6 steps/hour).
        horizon (int, optional): The number of future time steps to predict (default: 432, equivalent to 3 days at 10-minute intervals).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X (np.ndarray): The past data with shape `(lookback, num_features)`, excluding `log_density`(target value).
            - y (np.ndarray): The future values of `log_density` with shape `(horizon,)`.

    Raises:
        ValueError: If `T0` is not found in the DataFrame index or if there is insufficient historical data.
    """
    try:
        end_of_history = df.index.get_loc(T0)
    except KeyError:
        raise ValueError(f"T0 timestamp {T0} not found in DataFrame index.")

    if isinstance(end_of_history, slice):
        end_of_history = end_of_history.stop

    start_of_history = end_of_history - lookback
    if start_of_history < 0:
        raise ValueError("Not enough data for the specified lookback period.")

    df_past = df.iloc[start_of_history:end_of_history]
    df_future = df.iloc[end_of_history:end_of_history + horizon]

    # Extract past features (excluding 'log_density') and future target values
    X = df_past.drop(columns=['log_density']).to_numpy()
    y = df_future['log_density'].to_numpy()

    return X, y

def load_initial_states(path: str = "", full: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Loads the initial state dataset from multiple CSV files.

    This function reads multiple CSV files containing initial states for space weather data,
    concatenates them into a single DataFrame, and optionally samples a subset for validation.

    Args:
        path (str, optional): Path to dataset
        full (bool, optional): Whether to load the full dataset. 
            - If `True`, returns the full dataset.
            - If `False`, returns a sampled 10% validation dataset and the remaining dataset. 
            - Default is `False`.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]: A tuple containing:
            - df_remaining (pd.DataFrame): The full dataset (if `full=True`) or the dataset after sampling.
            - df_sampled (Optional[pd.DataFrame]): A sampled validation subset (if `full=False`), otherwise `None`.
    """
    combined_df = pd.DataFrame()
    
    initial_states_paths = [
        f"{path}00000_to_02284-initial_states.csv",
        f"{path}02285_to_02357-initial_states.csv",
        f"{path}02358_to_04264-initial_states.csv",
        f"{path}04265_to_05570-initial_states.csv", 
        f"{path}05571_to_05614-initial_states.csv",
        f"{path}05615_to_06671-initial_states.csv",
        f"{path}06672_to_08118-initial_states.csv",
    ]

    # Load and concatenate all initial state CSV files
    for path in initial_states_paths:
        df_init = pd.read_csv(
            path, 
            usecols=[
            'File ID', 'Timestamp', 'Semi-major Axis (km)', 'Eccentricity',
            'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)',
            'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)',
            'Altitude (km)'
            ],
            parse_dates=['Timestamp']
        )
        df_init = clean_initial_states(df_init)  # Apply cleaning function
        combined_df = pd.concat([combined_df, df_init], ignore_index=True)

    if not full:
        # Sample 10% of the rows with a fixed random seed for consistency
        df_sampled = combined_df.sample(frac=0.10, random_state=42)
        df_remaining = combined_df.drop(df_sampled.index)  # Remove sampled rows from main dataset
    else:
        df_remaining = combined_df
        df_sampled = None

    return df_remaining, df_sampled

def load_data(
    path: str = "",
    num_files: int = 8118,
    full: bool = False,
    val: bool = False,
    include_goes: bool = False,
    add_lags: bool = False,
    omni2_fields: List[str] = [],
    add_flags: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and processes multiple space weather data files with a progress bar.

    This function loads and processes space weather data files, applies feature 
    transformations such as lagged variables and missing data flags, and returns 
    the processed feature matrix and target variable.

    Args:
        path (str, optional): Path to dataset directory
        num_files (int, optional): Maximum number of files to process. Defaults to 8118.
        full (bool, optional): Whether to use the full dataset (8118 files). Defaults to False.
        val (bool, optional): Whether to use the validation dataset (812 files max). Defaults to False.
        include_goes (bool, optional): Whether to include GOES satellite data. Defaults to False.
        add_lags (bool, optional): Whether to include lagged features in the dataset. Defaults to False.
        omni2_fields (List[str], optional): List of OMNI2 dataset fields to include. Defaults to an empty list.
        add_flags (bool, optional): Whether to add missing data flags. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X_all: Feature matrix with shape (num_samples, lookback, num_features).
            - y_all: Target variable (log_density) with shape (num_samples, horizon).
    """
    FULL_DATA_SIZE = 8118
    # Adjust num_files based on dataset selection
    if val:
        num_files = min(num_files, round(FULL_DATA_SIZE * 0.1))  # Ensure no more than 812 files for validation
    elif full:
        num_files = FULL_DATA_SIZE  # Use full dataset
    else:
        num_files = min(num_files, round(FULL_DATA_SIZE * 0.9))  # Default behavior (training set)

    print("\n\nLoading Data...")

    # Load initial state and validation data
    df_init, val_init = load_initial_states(path=path, full=full)

    count = 0
    X_all, y_all = [], []
    nonusable_files = []
    df = val_init if val else df_init

    with tqdm(total=num_files, desc="Processing files", unit="file") as pbar:
        for _, row in df.iterrows():
            if count >= num_files:
                break

            file_id = int(row['File ID'])
            row, df_omni, df_density, df_goes, goes_is_missing = load_data_for_file_id(
                path, file_id, df, include_goes, omni2_fields=omni2_fields
            )

            # Process and unify time-series data
            df_10min = unify_time_series(
                row, df_omni, df_density, df_goes, goes_is_missing, 
                add_lags=add_lags, add_flags=add_flags
            )

            try:
                X, y = create_single_sample_ts(df_10min, row['Timestamp'])
            except ValueError:
                nonusable_files.append(f"File_{file_id}")
                pbar.update(1)
                continue

            # Ensure valid feature and target dimensions
            if np.isnan(X).any():
                print("Nans in X")
                nonusable_files.append(f"File_{file_id}")
                pbar.update(1)
                continue
            elif np.isnan(y).any() or len(y) != 432:
                nonusable_files.append(f"File_{file_id}")
                pbar.update(1)
                continue
            
            X_all.append(X)
            y_all.append(y)
            count += 1
            pbar.update(1)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"\nSkipped {len(nonusable_files)} files due to missing/invalid data")
    print(f"Final dataset shapes - X_all: {X_all.shape}, y_all: {y_all.shape}\n\n")
    
    return X_all, y_all
