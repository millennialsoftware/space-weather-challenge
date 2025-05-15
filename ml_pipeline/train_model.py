"""
train_model.py

This script trains, evaluates, and generates predictions using ml models for 
satellite density forecasting. It loads space weather data, preprocesses it, trains a 
model, evaluates its performance, and generates a submission file to test if it outperforms
the previous best model.

Main Steps:
    1. Load and preprocess satellite weather data.
    2. Split data into training, testing, and validation sets based on specifications.
    3. Scale features and target variables.
    4. Train a neural network model.
    5. Evaluate model performance on test and validation sets.
    6. Generate a JSON submission file with predictions.

Usage:
    This script is designed to be run as a module for training and evaluating 
    a ml model for satellite density prediction.
"""

# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# Model imports
from models.CNN_RNN import CNN_RNN
from models.CNN_RNN2 import CNN_RNN2
from models.CNN_Transformer import CNN_Transformer
from models.Van_Transformer import VanillaTransformer
from models.LSTM_Tranformer import LSTM_Transformer
from models.CNN_RNN_transformer import cnn_lstm_gru_transformer
from models.CNN_RNN_Transformer_Att import cnn_lstm_gru_transformer_selfAtt
from models.CNN_RNN_TransfromerEncoder import cnn_lstm_gru_transformerEncoder


# Supress deep learning framework logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Best params found from feature search
omni2_fields = [
    "Timestamp",
    "f10.7_index",
    "R_Sunspot_No",
    "YEAR",
    "DOY",
    "Hour",
    "Bartels_rotation_number",
    "num_points_IMF_averages",
    "num_points_Plasma_averages",
    "BX_nT_GSE_GSM",
    "BY_nT_GSE",
    "BZ_nT_GSE",
    "BY_nT_GSM",
    "BZ_nT_GSM",
    "Lat_Angle_of_B_GSE",
    "Long_Angle_of_B_GSE",
    "RMS_magnitude_nT",
    "RMS_field_vector_nT",
    "RMS_BX_GSE_nT",
    "RMS_BY_GSE_nT",
    "RMS_BZ_GSE_nT",
    "SW_Plasma_Temperature_K",
    "SW_Proton_Density_N_cm3",
    "SW_Plasma_Speed_km_s",
    "SW_Plasma_flow_long_angle",
    "SW_Plasma_flow_lat_angle",
    "sigma_T_K",
    "sigma_n_N_cm3",
    "sigma_V_km_s",
    "sigma_phi_V_degrees",
    "sigma_theta_V_degrees",
    "Kp_index",
    "Dst_index_nT",
    "ap_index_nT",
    "pc_index",
    "AE_index_nT",
    "AL_index_nT",
    "AU_index_nT",
    "Flow_pressure"
]

# Dictionary mapping numbers to (string representation, class reference)
models = {
    1: ("CNN_RNN", CNN_RNN),
    2: ("CNN_RNN2", CNN_RNN2),
    3: ("CNN_Transformer", CNN_Transformer),
    4: ("Van_Transformer", VanillaTransformer),
    5: ("LSTM Transformer", LSTM_Transformer),
    6: ("CNN_RNN_transformer", cnn_lstm_gru_transformer),
    7: ("CNN_RNN_Transfromer_Att", cnn_lstm_gru_transformer_selfAtt),
    8: ("CNN_RNN_TransfromerEncoder", cnn_lstm_gru_transformerEncoder )
}

# Display available models
print("\n\n\n")
for key, (name, _) in models.items():
    print(f"{key}: {name}")

selection = None

# Get model input from user
while selection not in models:
    try:
        selection = int(input("Select a model to train: "))
    except ValueError:
        print("Invalid input! Please enter a number.")

# If model is tranformer. Store file name to later fetch the custom tranformer block for submission
if selection in [3, 4, 5, 6, 7, 8, 9]:
    transformer = models[selection][0]
else:
    transformer = None

# Set the model class
ModelClass = models[selection][1]

# Set number of files
num_files = 1000000

# Preprocessing parameter flags
include_goes = False
add_lags = False
add_flags = True

# Use model selection's best params if set to true
use_best_params = False

# use all data for training; don't hold out a validation data set
full = True

# Dont even hold out test set
test_set = False

if test_set:
    full = True

# Validation set size
if full:
    num_val_files = 0
else:
    num_val_files = num_files // 10

# Load environment variables
load_dotenv()

# Get data path
PATH = os.getenv("DATA_PATH")

# Load training and validation sets
X_all, y_all = load_data(path=PATH, num_files=num_files, full=full, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)
X_val, y_val = load_data(path=PATH, num_files=num_val_files, val=True, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)

# There should be no validation set if using all data for training
if full:
    X_val = None
    y_val = None

if test_set:
    X_train = X_all
    y_train = y_all
    X_test = None
    y_test = None
else:
    # Split training into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

# Scale data 

# Set binary flags to be ignored from scaling 
num_feats = X_train.shape[2]  # Total number of features

num_flags = 0

# Check if add_flags is True
if add_flags:
    num_flags += 1  
    if include_goes:
        num_flags += 1

# Set start_flag_index only if previous conditions are met
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

# Load in best params if true
if use_best_params:
    try:
        with open("models/best_params.json", "r") as file:
            best_param_dict = json.load(file)
            best_params = best_param_dict.get(models[selection][0])
            print(f"Using params: {best_params}\n\n")
    except FileNotFoundError:
        best_params = None
else:
    best_params = None

# Use an empty dictionary if best_params is None
kwargs = best_params if best_params is not None else {}

# Initialize model with base arguments and any additional best parameters
nn_model = ModelClass(
    input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
    horizon=y_train.shape[1],
    **kwargs
)

# Compile model
model = nn_model.get_model()

# Print model summary
model.summary()

validation_data = None
if not test_set:
    validation_data = (X_test_scaled, y_test_scaled)

# Fit the model on training data
model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=validation_data,
    epochs=nn_model.epochs,
    batch_size=nn_model.batch_size,
    callbacks=nn_model.get_callbacks()
)

if not test_set:
    # Test the model on test data
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)

    # Undo transformations
    y_pred = undo_transformation(scaler_y, y_pred_scaled)
    y_test = undo_transformation(scaler_y, y_test_scaled)

    # Calculate rMSE, MAE, and sMAPE for predictions
    print("\n\n\n\033[32mTest Metrics: ")
    calculate_metrics(model, y_test, y_pred, save=False)


if not full:
    # Test the model on validation data
    val_pred_scaled = model.predict(X_val_scaled, verbose=0)
    val_pred = undo_transformation(scaler_y, val_pred_scaled)
    y_val = undo_transformation(scaler_y, y_val_scaled)

    # Calculate rMSE, MAE, and sMAPE for predictions
    print("\n\n\n\033[32mValidation Metrics: ")
    calculate_metrics(model, y_val, val_pred, save=False)

# Generate submission json file
generate_submission(
    selection=models[selection][0], path=PATH, model=model, X_scaler=scaler_x, y_scaler=scaler_y, 
    include_goes=include_goes, add_lags=add_lags, add_flags=add_flags, 
    omni2_fields=omni2_fields, best_submission_path=None, transformer=transformer
)