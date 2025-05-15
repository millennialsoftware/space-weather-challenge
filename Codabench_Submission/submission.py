"""
Satellite Density Prediction Pipeline.

This script loads a trained machine learning model and associated scalers, processes 
space weather and orbital datasets, and generates predictions for satellite orbital 
density over a three-day forecast period. It saves the results in a JSON file.

Workflow:
    1. Loads trained model and feature scalers.
    2. Reads and cleans the initial states dataset.
    3. Iterates through each file ID, loading and processing time-series data.
    4. Scales features and predicts orbital density using the trained model.
    5. Saves predictions to a JSON file.

Functions:
    - load_data_for_file_id: Loads space weather data for a given file ID.
    - unify_time_series: Merges and aligns time-series data into a common format.
    - create_single_sample_ts: Extracts time-series samples for prediction.
    - scale_features: Scales numerical features while preserving binary flags.
"""

# Imports
import pandas as pd
from datetime import timedelta
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
from features import clean_initial_states
from load_data import *
import time
from tensorflow.keras.utils import custom_object_scope

# Initialize time object to track submission length
start_time = time.time() 

# Define absolute paths
TEST_DATA_DIR = Path('/app/data/dataset/test')
TEST_PREDS_FP = Path('/app/output/prediction.json')

# Set paths for initial state file on server
initial_states_file = os.path.join('/app/input_data',"initial_states.csv")

# Load in scalers
with open("/app/ingested_program/X_scaler.pkl", "rb") as f:
    X_scaler = pickle.load(f)

with open("/app/ingested_program/y_scaler.pkl", "rb") as f:
    y_scaler = pickle.load(f)

# Load in parameters

# Define the file path
parameters_path = "/app/ingested_program/parameters.json"

# Load parameters from the JSON file
if os.path.exists(parameters_path):
    with open(parameters_path, "r") as param_file:
        parameters = json.load(param_file)

    # Extract values
    add_flags = parameters.get("add_flags", False)
    add_lags = parameters.get("add_lags", False)
    include_goes = parameters.get("include_goes", False)
    omni2_fields = parameters.get("omni2_fields", [])
    transformer = parameters.get("transformer", None)
    print("Loaded parameters:", parameters)
else:
    raise FileNotFoundError("Paramaters file not found")

# Load in trained model. Handle custom transformers different
if transformer is not None:
    # import classes from custom transformer model
    from transformer import *
    
    # It will always (or should always) have a transformer block 
    custom_objects = {'TransformerBlock': TransformerBlock}
    # Check if other custom classes exist before adding it.
    if 'SelfAttention' in globals():
        custom_objects['SelfAttention'] = SelfAttention
    if 'PositionalEmbedding' in globals():
        custom_objects['PositionalEmbedding'] = PositionalEmbedding
    #load model with custom objects
    with custom_object_scope(custom_objects):
        model = load_model("/app/ingested_program/model.keras")
else:
    model = load_model("/app/ingested_program/model.keras")

# Print model summary
model.summary()

# Load in scalers
with open("/app/ingested_program/X_scaler.pkl", "rb") as f:
    X_scaler = pickle.load(f)

with open("/app/ingested_program/y_scaler.pkl", "rb") as f:
    y_scaler = pickle.load(f)

# Print start of data loading
print("\nStarting Predictions\n")

# Load in initial-states file
df_init = pd.read_csv(
    initial_states_file,
    usecols=[
        'File ID', 'Timestamp', 'Semi-major Axis (km)', 'Eccentricity',
        'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)',
        'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)',
        'Altitude (km)'
    ],
    parse_dates=['Timestamp']
)

# Convert Altitudes for test data to kilometers if they are in meters
mask = df_init['Altitude (km)'] >= 100000
df_init.loc[mask, 'Altitude (km)'] = df_init.loc[mask, 'Altitude (km)'] / 1000

# Show number of initial states
num_states = len(df_init)
print(f"Number of initial states{num_states}")

# Clean initial states file
df_init = clean_initial_states(df_init)

# Show if there is any nan values after cleaning
print("NaN values per column in initial state:")
print(df_init.isna().sum())

# Counter to keep track of how many files are left
file_count = 1

# Create empty dictionary for predictions
predictions = {}

# Iterate through each row, generating predictions for each file id.
for _, row in df_init.iterrows():
   
    # Get prediction interval
    initial_time = pd.to_datetime(row['Timestamp']).ceil('10min')
    end_time = initial_time + timedelta(days=2, hours=23, minutes=50)

    # Generate a range of timestamps every 10 minutes for the prediction interval
    timestamps = pd.date_range(start=initial_time, end=end_time, freq='10min')

    # Create a predictions DataFrame with the generated timestamps
    predictions_df = pd.DataFrame({'Timestamp': timestamps})    

    # Set the file id
    file_id = int(row['File ID'])

     # Load in data for a file id
    row, df_omni, df_goes, goes_is_missing = load_data_for_file_id(TEST_DATA_DIR,
        file_id, df_init, include_goes, omni2_fields=omni2_fields
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
    y_pred_scaled = model.predict(X_input)

    # Check if nans are in output
    if np.isnan(y_pred_scaled).sum() > 0:
        print("NaNs found in predictions!")

    # Undo Min-Max Scaling for predicted values
    y_pred_scaled_flat = y_pred_scaled.reshape(-1, 1)
    y_pred_flat = y_scaler.inverse_transform(y_pred_scaled_flat)

    # Undo the log transformation for predicted values
    y_pred = np.exp(y_pred_flat)
    y_pred = np.exp(y_pred_flat)

    # Add predictions to prediction dataframe
    predictions_df['Orbit Mean Density (kg/m^3)'] = y_pred.flatten()

    # Add predictions to dictionary 
    predictions[file_id] = {
        "Timestamp": list(map(lambda ts: ts.isoformat(), predictions_df["Timestamp"])),
        "Orbit Mean Density (kg/m^3)": predictions_df['Orbit Mean Density (kg/m^3)'].tolist(),
    }

    # Print that model finished for a file
    print(f"\nModel execution for {file_id} Finished\n")
    print(f"\nFiles left: {num_states - file_count}\n")
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds\n")

    # Show first prediction to confirm functionality
    if file_count == 1:
        print(predictions_df)
    file_count += 1

# Save the predictions to a JSON file
with open(TEST_PREDS_FP, "w") as outfile:
    json.dump(predictions, outfile)

# Confirm predictions saved
print(f"\nSaved predictions to: {TEST_PREDS_FP}\n")
end_time = time.time()  # End timer
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")