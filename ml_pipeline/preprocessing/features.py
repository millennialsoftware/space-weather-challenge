"""
features.py

This module contains functions to clean and preprocess different datasets 
related to space weather. Each function applies:
- Removal of out-of-range values by replacing them with NaN.
- Interpolation to fill missing values.
- Forward-fill and backward-fill techniques for handling remaining NaNs.

Functions:
- clean_initial_states: Cleans initial state dataset (e.g., satellite data).
- clean_omni2: Cleans and preprocesses OMNI2 space weather data.
- clean_sat_density: Cleans satellite density measurements.
- clean_goes: Cleans GOES satellite data.
- compute_solar_zenith: Compute the solar zenith angle for timesteps.

Typical usage example:

    from features import clean_omni2, clean_initial_states

    df_omni = pd.read_csv("omni2.csv")
    df_omni = clean_omni2(df_omni)

    df_init = pd.read_csv("initial_states.csv")
    df_init = clean_initial_states(df_init)
"""

# Imports
from typing import Union
import numpy as np
import pandas as pd
from scipy.signal import welch

def clean_initial_states(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the initial states dataset.

    This function performs the following steps:
    1. Replaces out-of-range values with NaN based on valid ranges.
    2. Interpolates missing values to maintain data continuity.
    3. Applies backward-fill and forward-fill to handle any remaining NaNs.

    Args:
        df (pd.DataFrame): The initial states dataset.

    Returns:
        pd.DataFrame: A cleaned version of the original dataset, with:
            - Out-of-range values replaced with NaN.
            - Missing values interpolated.
            - Remaining NaNs filled using forward-fill and backward-fill.
    """

    # Define valid ranges for each feature
    valid_ranges = {
        "Latitude (deg)": (-90, 90),
        "Longitude (deg)": (-180, 180),
        "Altitude (km)": (0, 1000),
        "Semi-major Axis (km)": (0, 10000),
        "Eccentricity": (0, 1),
        "Inclination (deg)": (0, 180),
        "Argument of Perigee (deg)": (0, 360),
        "RAAN (deg)": (0, 360),
        "True Anomaly (deg)": (0, 360)
    }

    # Replace out-of-range values with NaN
    for column, (min_val, max_val) in valid_ranges.items():
        if column in df.columns:
            df[column] = df[column].astype(float)
            df[column] = df[column].where(
                (df[column] >= min_val) & (df[column] <= max_val), np.nan
            )

    # Identify numeric columns to interpolate (excluding 'File ID' and 'Timestamp')
    columns_to_interpolate = [
        col for col in df.columns if col not in ['File ID', 'Timestamp']
    ]

    # Convert selected columns to float and apply interpolation
    df[columns_to_interpolate] = df[columns_to_interpolate].astype(float)
    df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear')

    # Apply backward-fill and forward-fill for remaining NaNs
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df

def clean_omni2(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses an OMNI2 data file.

    This function performs the following steps:
    1. Replaces out-of-range values with NaN based on valid ranges.
    2. Interpolates missing values using linear interpolation.
    3. Applies backward-fill and forward-fill to handle remaining NaNs.

    Args:
        df (pd.DataFrame): The OMNI2 data file containing space weather parameters.

    Returns:
        pd.DataFrame: A cleaned version of the file with:
            - Invalid values replaced with NaN.
            - Missing values interpolated.
            - Remaining NaNs filled using forward-fill and backward-fill.
    """

    # Define valid value ranges for each column
    valid_ranges = {
        "YEAR": (1957, 2100),
        "DOY": (1, 366),
        "Hour": (0, 23),
        "Bartels_rotation_number": (2000, 3000),
        "num_points_IMF_averages": (1, 100),
        "num_points_Plasma_averages": (1, 100),

        # Magnetic Field Components (nT)
        "Scalar_B_nT": (0, 100), 
        "Vector_B_Magnitude_nT": (0, 100),
        "Lat_Angle_of_B_GSE": (-90, 90),
        "Long_Angle_of_B_GSE": (0, 360),
        "BX_nT_GSE_GSM": (-100, 100),
        "BY_nT_GSE": (-100, 100), 
        "BZ_nT_GSE": (-100, 100), 
        "BY_nT_GSM": (-100, 100), 
        "BZ_nT_GSM": (-100, 100),
        "RMS_magnitude_nT": (0, 100),
        "RMS_field_vector_nT": (0, 100),
        "RMS_BX_GSE_nT": (0, 100),
        "RMS_BY_GSE_nT": (0, 100),
        "RMS_BZ_GSE_nT": (0, 100),

        # Solar Wind Plasma Parameters
        "SW_Plasma_Temperature_K": (1e3, 1e6),
        "SW_Proton_Density_N_cm3": (0.1, 100),
        "SW_Plasma_Speed_km_s": (0, 2500),
        "SW_Plasma_flow_long_angle": (-180, 180),
        "SW_Plasma_flow_lat_angle": (-90, 90),
        "Alpha_Prot_ratio": (0, 0.5),

        # Standard Deviations of Plasma Parameters
        "sigma_T_K": (0, 1e5),
        "sigma_n_N_cm3": (0, 50),
        "sigma_V_km_s": (0, 100),
        "sigma_phi_V_degrees": (0, 45),
        "sigma_theta_V_degrees": (0, 45),
        "sigma_ratio": (0, 1),

        # Solar Wind Pressure & Derived Parameters
        "Flow_pressure": (0, 50),
        "E_electric_field": (-50, 50), 
        "Plasma_Beta": (0, 100),
        "Alfen_mach_number": (0, 100), 
        "Magnetosonic_Mach_number": (0, 20), 
        "Quasy_Invariant": (0, 1),

        # Geomagnetic Indices
        "Kp_index": (0,90),
        "Dst_index_nT": (-500, 100),
        "ap_index_nT": (0, 400),
        "AE_index_nT": (0, 5000), 
        "AL_index_nT": (-2500, 10), 
        "AU_index_nT": (-500, 500),
        "pc_index": (-50, 50),
       
        # Solar Activity & Proton Flux
        "f10.7_index": (0, 500), 
        "R_Sunspot_No": (0, 400), 
        "Lyman_alpha": (-50, 50),
        "Proton_flux_>1_Mev": (0, 500),
        "Proton_flux_>2_Mev": (0, 500),
        "Proton_flux_>4_Mev": (0, 500),
        "Proton_flux_>10_Mev": (0, 500),
        "Proton_flux_>30_Mev": (0, 500),
        "Proton_flux_>60_Mev": (0, 500),
    }

    # Replace out-of-range values with NaN
    for column, (min_val, max_val) in valid_ranges.items():
       if column in df.columns:
            df[column] = df[column].astype(float)
            df[column] = df[column].where(
                (df[column] >= min_val) & (df[column] <= max_val), np.nan
            )

    
    # Apply interpolation for missing values - only numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    # Apply backward-fill and forward-fill for any remaining NaNs
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    return df

def clean_sat_density(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses a satellite density data file.

    This function performs the following steps:
    1. Replaces out-of-range values with NaN based on valid density ranges.
    2. Interpolates missing values using linear interpolation.
    3. Applies backward-fill and forward-fill to handle remaining NaNs.

    Args:
        df (pd.DataFrame): The satellite density data file.

    Returns:
        pd.DataFrame: A cleaned version of the file with:
            - Out-of-range values replaced with NaN.
            - Missing values interpolated.
            - Remaining NaNs filled using forward-fill and backward-fill.
    """

    # Set valid range for density
    column = 'Orbit Mean Density (kg/m^3)'
    min_val = 1e-18
    max_val = 1e-8

    # Replace out-of-range values with NaN
    if column in df.columns:
        df[column] = df[column].where(
            (df[column] >= min_val) & (df[column] <= max_val), np.nan
        )

        # Apply interpolation if there are NaNs
        if df[column].isna().sum() > 0:
            df[column] = df[column].interpolate(method='linear')

        # Apply backward-fill and forward-fill
        df.bfill(inplace=True)
        df.ffill(inplace=True)
    else:
        print(f"Warning: Column '{column}' not found in DataFrame.")

    return df

def clean_goes(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses a GOES satellite data file.

    This function performs the following steps:
    1. Converts the 'Timestamp' column to datetime format and removes invalid rows.
    2. Replaces out-of-range values with NaN based on valid ranges.
    3. Interpolates missing values using linear interpolation.
    4. Applies backward-fill and forward-fill to handle remaining NaNs.

    Args:
        df (pd.DataFrame): The GOES satellite data file.

    Returns:
        pd.DataFrame: A cleaned version of the file with:
            - Invalid timestamps removed.
            - Out-of-range values replaced with NaN.
            - Missing values interpolated.
            - Remaining NaNs filled using forward-fill and backward-fill.
    """

    # Convert 'Timestamp' column to datetime, removing invalid rows
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)


    # Define valid ranges for GOES data
    valid_ranges = {"xrsa_flux": (1e-20, 1e-1), "xrsb_flux": (1e-20, 1e-1)}

   # Replace out-of-range values with NaN
    for column, (min_val, max_val) in valid_ranges.items():
       if column in df.columns:
            df[column] = df[column].astype(float)
            df[column] = df[column].where(
                (df[column] >= min_val) & (df[column] <= max_val), np.nan
            )

       

    # Apply interpolation to fill in missing values
    df.interpolate(method="linear", inplace=True)

    # Apply backward-fill and forward-fill
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df


import numpy as np
import pandas as pd

def compute_solar_zenith(
    times: pd.DatetimeIndex,
    lat: Union[pd.Series, np.ndarray],
    lon: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Compute the solar zenith angle (in degrees) for each timestamp and location.
    
    Args:
        times:    pandas.DatetimeIndex of UTC timestamps.
        lat:      Array or Series of latitudes (degrees).
        lon:      Array or Series of longitudes (degrees).
        
    Returns:
        numpy array of solar zenith angles (degrees).
    """
    # Ensure arrays
    times = pd.to_datetime(times)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    
    # Fractional day of year and minutes since midnight
    doy  = times.dayofyear.values.astype(float)
    mins = times.hour.values * 60 + times.minute.values + times.second.values / 60.0

    # Fractional year (radians)
    γ = 2 * np.pi / 365 * (doy - 1 + mins / 1440)

    # Solar declination δ (radians)
    δ = (
        0.006918
        - 0.399912 * np.cos(γ)
        + 0.070257 * np.sin(γ)
        - 0.006758 * np.cos(2 * γ)
        + 0.000907 * np.sin(2 * γ)
        - 0.002697 * np.cos(3 * γ)
        + 0.00148  * np.sin(3 * γ)
    )

    # Equation of Time (minutes)
    EoT = 229.18 * (
        0.000075
        + 0.001868 * np.cos(γ)
        - 0.032077 * np.sin(γ)
        - 0.014615 * np.cos(2 * γ)
        - 0.040849 * np.sin(2 * γ)
    )

    # True Solar Time (minutes)
    # Assume timestamps are UTC; timezone offset = 0
    TST = mins + EoT + 4 * lon

    # Hour angle H (radians)
    H = np.deg2rad((TST / 4) - 180)

    # Convert latitude to radians
    φ = np.deg2rad(lat)

    # Cosine of solar zenith
    cos_z = np.sin(φ) * np.sin(δ) + np.cos(φ) * np.cos(δ) * np.cos(H)
    cos_z = np.clip(cos_z, -1.0, 1.0)

    # Zenith angle (degrees)
    zenith = np.rad2deg(np.arccos(cos_z))
    return zenith