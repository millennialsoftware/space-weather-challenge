"""
feature_exhaustion.py

This script performs feature exhaustion to evaluate the best feature combination to use for
a neural network model trained for space weather prediction. It systematically tests feature subsets 
from the OMNI2 dataset and logs the model performance.

The script follows these steps:
1. Loads and preprocesses the dataset using different feature combinations.
2. Splits the dataset into training, validation, and test sets.
3. Scales the features and target variables.
4. Trains a CNN_Transformer-based deep learning model on each feature subset.
5. Evaluates model performance based on propagation scores.
6. Identifies the best-performing feature subset and updates a JSON file.
7. Sends an email update with results.
8. Saves the best predictions and updates the dataset for future iterations.

Usage:
    Run the script to automatically iterate over feature subsets, train models, 
    evaluate performance, and update the feature selection dictionary.

Functions:
    get_combinations(groups):
        Generates all possible feature combinations from predefined groups.

    send_email(recipients, subject, content):
        Sends an email notification with results.

    generate_submission(path, model, X_scaler, y_scaler, include_goes, add_lags, add_flags, omni2_fields, best_submission_path):
        Generates and evaluates model predictions for a given feature subset.

    scale_features(scaler, X_train, X_test, X_val, start_flag_index):
        Scales input features while preserving binary flags.

    scale_targets(scaler, y_train, y_test, y_val):
        Scales target values for training.

    CNN_Transformer(input_shape, horizon, **kwards):
        Initializes and returns a CNN-Transformer-based deep learning model.

Notes:
    - The script modifies `field_combinations.json` to track the best feature subset.
    - If `field_combinations.json` is missing, it regenerates feature combinations.
"""

# Imports
import os
import tensorflow as tf
import logging
from submission_generator import generate_submission
from metrics import calculate_metrics
from preprocessing.load_data import load_data
from preprocessing.scaling import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from dotenv import load_dotenv
import json
from email_services import send_email
from itertools import chain, combinations
import json
from typing import Dict, List

# Model import - Using transformer to reduce training time
from models.CNN_Transformer import CNN_Transformer

# Supress deep learning framework logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Omni2 features grouped into categories to generate exhausitve feature grid
groups = {
    "Time-Based Fields": ["YEAR", "DOY", "Hour", "Bartels_rotation_number"],
    "Data Quality & Sampling Statistics": ["num_points_IMF_averages", "num_points_Plasma_averages"],
    "Magnetic Field Properties 1": [
        "Scalar_B_nT", "Vector_B_Magnitude_nT"
    ],
    "Magnetic Field Properties 2": [
        "BX_nT_GSE_GSM", "BY_nT_GSE", "BZ_nT_GSE", "BY_nT_GSM", "BZ_nT_GSM", "Lat_Angle_of_B_GSE", "Long_Angle_of_B_GSE"
    ],
    "Magnetic Field Properties 3": [
        "RMS_magnitude_nT", "RMS_field_vector_nT", "RMS_BX_GSE_nT", "RMS_BY_GSE_nT", "RMS_BZ_GSE_nT"
    ],
    "Plasma Properties 1": [
        "SW_Plasma_Temperature_K", "SW_Proton_Density_N_cm3", "SW_Plasma_Speed_km_s",
        "SW_Plasma_flow_long_angle", "SW_Plasma_flow_lat_angle"
    ],
    "Plasma Properties 2": [
        "sigma_T_K", "sigma_n_N_cm3", "sigma_V_km_s", "sigma_phi_V_degrees",
        "sigma_theta_V_degrees"
    ],
    "Solar Wind & Space Weather Indices": [
        "Flow_pressure", "E_electric_field", "Plasma_Beta", "Alfen_mach_number", "Magnetosonic_Mach_number",
        "Quasy_Invariant"
    ],
    "Geomagnetic Indices 1": [
        "Kp_index", "Dst_index_nT", "ap_index_nT", "pc_index"
    ],
    "Geomagnetic Indices 2": [
        "AE_index_nT", "AL_index_nT", "AU_index_nT"
    ],
    "High Energy Proton Flux": [
        "Proton_flux_>1_Mev", "Proton_flux_>2_Mev", "Proton_flux_>4_Mev", "Proton_flux_>10_Mev",
        "Proton_flux_>30_Mev", "Proton_flux_>60_Mev"
    ]
}

def get_combinations(groups: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """Generates all possible feature combinations from predefined groups.

    Args:
        groups (Dict[str, List[str]]): A dictionary mapping feature categories to lists of related features.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing:
            - "combinations" (Dict[str, List[str]]): Keys as iteration identifiers and values as feature subsets.
            - "best_features" (Dict[str, List[str]]): Placeholder for tracking the best-performing feature subset.
    """
    all_combinations = []
    
    for subset in chain.from_iterable(combinations(groups.keys(), r) for r in range(1, len(groups) + 1)):
        selected_fields = ["Timestamp", "f10.7_index", "R_Sunspot_No"] + list(chain.from_iterable(groups[group] for group in subset))
        all_combinations.append(selected_fields)
    
    return {"combinations": {f"iteration_{i}": combination for i, combination in enumerate(all_combinations)}, "best_features": {}}

# Load in field combinations json file - if it doesn't exist generate new dictionary
try:
    with open("models/field_combinations.json", "r") as json_file:
        features_dict = json.load(json_file)
except FileNotFoundError:
    features_dict = get_combinations(groups)

# Use 3000 files
num_files = 3000

# Preprocessing parameters
include_goes = False
add_lags = False
add_flags = True

# Validation set size
num_val_files = num_files // 10

# Load environment variables
load_dotenv()

# Set path to dataset
PATH = os.getenv("DATA_PATH")

# Iterate through a copy of the possible combinations
for iteration, omni2_fields in list(features_dict["combinations"].items()):

    # Load data using the fields in the current iteration
    X_all, y_all = load_data(path=PATH, num_files=num_files, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)
    X_val, y_val = load_data(path=PATH, num_files=num_val_files, val=True, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)

    # Split training into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, 
        y_all, 
        test_size=0.1,    
        random_state=42,  
        shuffle=True      
    )

    # Set binary flags to be ignored from scaling 
    num_feats = X_train.shape[2]

    num_flags = 0

    # Check if add_flags is True
    if add_flags:
        num_flags += 1  
        if include_goes:
            num_flags += 1

    # Set start_flag_index only if conditions are met
    if add_flags and num_flags > 0:
        start_flag_index = num_feats - num_flags
    else:
        start_flag_index = None

    # Initialize the StandardScaler for feature scaling
    scaler_x = StandardScaler()

    # Scale Features
    X_train_scaled, X_test_scaled, X_val_scaled = scale_features(scaler_x, X_train, X_test, X_val, start_flag_index=start_flag_index)

    # Initialize the MinMaxScaler for target scaling
    scaler_y = MinMaxScaler()

    # Scale Targets
    y_train_scaled, y_test_scaled, y_val_scaled = scale_targets(scaler_y, y_train, y_test, y_val)

    # Results dictionary
    results = {}

    # Boolean to hold if it is best performing so far
    isBest = False

    # Fit the model 3 times, recoding the results
    for i in range(3):

        # Initialize model - only 50 epochs for quicker training
        nn_model = CNN_Transformer(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            horizon=y_train.shape[1],
            epochs=100
        )

        # Compile model
        model = nn_model.get_model()

        # Print model summary
        model.summary()

        # Fit the model on training data
        model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_test_scaled, y_test_scaled),
            epochs=nn_model.epochs,
            batch_size=nn_model.batch_size,
            callbacks=nn_model.get_callbacks()
        )

        # Generate submission and get propagation score as well as predictions dictionary
        ps, predictions = generate_submission(path=PATH, model=model, X_scaler=scaler_x, y_scaler=scaler_y, include_goes=include_goes, add_lags=add_lags, add_flags=add_flags, omni2_fields=omni2_fields, best_submission_path="submissions/best_features.json")

        # Add results to dictionary
        results[f"iteration_{i}"] = {"ps": ps, "predictions": predictions}

    # Check how many times model performed better
    positive_ps_count = len([ps for iteration in results.values() if (ps := iteration["ps"]) >= 0])

    # Get the best iteration based on propagation score
    best_iteration = max(results.items(), key=lambda item: item[1]["ps"])

    # Check if model performed better majority of time
    if positive_ps_count >= 2:

        # Extract the predictions dictionary from the best iteration
        best_predictions = best_iteration[1]["predictions"]

        # Save the best predicitons to a json file
        with open("submissions/best_features.json", "w") as outfile:
            json.dump(predictions, outfile)

        # Update the dictionary to hold the best score and fields
        features_dict["best_features"]["ps"] = best_iteration[1]["ps"]
        features_dict["best_features"]["fields"] = omni2_fields
        
        # Set isBest flag to True
        isBest = True
    else:

        # Model did not perform better
        isBest = False

    # HTML content for update email
    email_content = f"""
    <html>
    <body>
        <h2 style="color:green; text-align: center;">Feature Exhaustion Update</h2>
        <p><b>Scores: </b> {results["iteration_0"]["ps"]}, {results["iteration_1"]["ps"]}, {results["iteration_2"]["ps"]}</p>
        <p><b>Best Score: </b> {best_iteration[1]["ps"]}</p>
        <p><b>Omni2 Fields: </b> {omni2_fields}</p>
        <hr>
        <p>Performed <span style="color:{"green" if isBest else "red"};"><b>{"better" if isBest else "worse"}</b></span> than previous best model</p>
    </body>
    </html>
    """

    # Send update email
    send_email(subject="Feature Exhaustion Update", content=email_content)

    # Remove tested iteration
    del features_dict["combinations"][iteration]

    # Save updated dictionary
    with open("models/field_combinations.json", "w") as json_file:
        json.dump(features_dict, json_file, indent=4)


