"""
feature_importance.py
"""

# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
from preprocessing.load_data import load_data
from preprocessing.scaling import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from dotenv import load_dotenv
import json
from tensorflow.keras.models import load_model
import pickle

# Model imports
from models.CNN_RNN import CNN_RNN
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt


# For loading trained model 
def load_model_with_retry(initial_path, max_attempts=3):
    """
    Attempts to load a model from the provided initial_path by appending "/model.keras" and retries on failure.

    This function will try to load the model from the given path. If the loading process fails due to
    an OSError, IOError, or ValueError, it prompts the user to provide an alternative path, and retries up to
    max_attempts times. If all attempts fail or the user cancels by entering an empty string, it raises a
    FileNotFoundError.

    Parameters:
        initial_path (str): The initial directory path where the model is expected to be located.
        max_attempts (int, optional): The maximum number of attempts to load the model. Defaults to 3.

    Returns:
        tuple: A tuple containing the loaded model and the path (str) from which it was successfully loaded.

    Raises:
        FileNotFoundError: If the model cannot be loaded after the specified number of attempts.
    """
    path = initial_path
    for attempt in range(1, max_attempts + 1):
        try:
            model = load_model(path+"/model.keras")
            print(f"Model loaded from: {path}")
            return model,path
        except (OSError, IOError,ValueError) as e:
            print(f"Attempt {attempt}: could not load model from '{path}': {e}")
            if attempt < max_attempts:
                path = input("Enter an alternative model path (or press Enter to cancel): ").strip()
                if not path:
                    break
    raise FileNotFoundError(f"Failed to load model after {max_attempts} attempts.")

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
    2: ("Load Pre-Trained Model", None )
}

# Display available models
print("\n\n\n")
for key, (name, _) in models.items():
    print(f"{key}: {name}")

selection = None

# Get model input from user
while selection not in models:
    try:
        selection = int(input("Select a model: "))
    except ValueError:
        print("Invalid input! Please enter a number.")



# None if we are loading in trained model
if selection ==2:
    ModelClass = None
else:
    # Set the model class
    ModelClass = models[selection][1]

# Set number of files
num_files = 1000

# Preprocessing parameter flags
include_goes = False
add_lags = False
add_flags = True

# Use model selection's best params if set to true
use_best_params = False

# use all data for training; don't hold out a validation data set
full = False

# Load environment variables
load_dotenv()

# Get data path
PATH = os.getenv("DATA_PATH")

# Load training and validation sets
X_all, y_all = load_data(path=PATH, num_files=num_files, full=full, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)

# Split training into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# set X and Y Val to none 
X_val=None 
y_val = None

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

# Handle diffrently for cases of loading trained model or training new model
if ModelClass is not None:

    # Initialize the StandardScaler for feature scaling
    scaler_x = StandardScaler()

    # Initialize the MinMaxScaler for target scaling
    scaler_y = MinMaxScaler()

    # Scale Features
    X_train_scaled, X_test_scaled, X_val_scaled = scale_features(scaler_x, X_train, X_test, X_val, start_flag_index=start_flag_index)

    # Scale Targets
    y_train_scaled, y_test_scaled, y_val_scaled = scale_targets(scaler_y, y_train, y_test, y_val)

    # Initialize model with base arguments and any additional best parameters
    nn_model = ModelClass(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        horizon=y_train.shape[1],
        **kwargs
    )
     # Compile model
    model = nn_model.get_model()
    # Fit the model on training data
    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=nn_model.epochs,
        batch_size=nn_model.batch_size,
        callbacks=nn_model.get_callbacks()
    )
    
else: 
    # get model/scalers directory 
    model_dir = input("Input directory to model and scalers -->  ")
    
    # load model with error handling
    model, model_dir = load_model_with_retry(model_dir)

    # Load in scalers
    with open(model_dir + "/X_scaler.pkl", "rb") as f:
        scaler_x = pickle.load(f)

    with open(model_dir+ "/y_scaler.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    # Scale Features
    X_train_scaled, X_test_scaled, X_val_scaled = scale_features(scaler_x, X_train, X_test, X_val, start_flag_index=start_flag_index)

    # Scale Targets
    y_train_scaled, y_test_scaled, y_val_scaled = scale_targets(scaler_y, y_train, y_test, y_val)


# Define Feature Names
features = [
    'YEAR',
    'Bartels_rotation_number',
    'num_points_IMF_averages',
    'num_points_Plasma_averages',
    'Lat_Angle_of_B_GSE',
    'Long_Angle_of_B_GSE',
    'BX_nT_GSE_GSM',
    'BY_nT_GSE',
    'BZ_nT_GSE',
    'BY_nT_GSM',
    'BZ_nT_GSM',
    'RMS_magnitude_nT',
    'RMS_field_vector_nT',
    'RMS_BX_GSE_nT',
    'RMS_BY_GSE_nT',
    'RMS_BZ_GSE_nT',
    'SW_Plasma_Temperature_K',
    'SW_Proton_Density_N_cm3',
    'SW_Plasma_Speed_km_s',
    'SW_Plasma_flow_long_angle',
    'SW_Plasma_flow_lat_angle',
    'sigma_T_K',
    'sigma_n_N_cm3',
    'sigma_V_km_s',
    'sigma_phi_V_degrees',
    'sigma_theta_V_degrees',
    'Flow_pressure',
    'Kp_index',
    'R_Sunspot_No',
    'Dst_index_nT',
    'ap_index_nT',
    'f10.7_index',
    'AE_index_nT',
    'AL_index_nT',
    'AU_index_nT',
    'pc_index',
    'Semi-major Axis (km)',
    'Eccentricity',
    'Inclination (deg)',
    'RAAN (deg)',
    'Argument of Perigee (deg)',
    'Latitude_dyn',
    'Longitude_dyn',
    'altitude_dyn',
    'anom_sin',
    'anom_cos',
    'lat_sin',
    'lat_cos',
    'lon_sin',
    'lon_cos',
    'lst_sin',
    'lst_cos',
    'solar_zenith',
    'sunlit_flag',
    'kp_diff_1h',
    'storm_phase',
    'Bz_neg',
    'Bz_pos',
    'Bz_neg_x_pressure',
    'speed_x_kp',
    'clock_sin',
    'clock_cos',
    'B_mag',
    'E_y',
    'epsilon',
    'doy_sin2',
    'doy_cos2',
    'hour_sin2',
    'hour_cos2',
    'Omni_Missing_Flag'
]

# set up score function to calulate MSE
def score(X3d, y_true_scaled):
    y_pred_scaled = model.predict(X3d)
    y_pred = undo_transformation(scaler_y, y_pred_scaled)
    y_true = undo_transformation(scaler_y, y_true_scaled)
    return mean_squared_error(
        y_true.flatten(),
        y_pred.flatten()
    )


# Baseline MSE score on the test set
base_score = score(X_test_scaled, y_test_scaled)
format_base_score = "{:e}".format(base_score)
print(f"Baseline test MSE: {format_base_score}")

# Permutation importance
n_test, T, F = X_test_scaled.shape
importances = np.zeros(F)
n_repeats = 3  # Number of times to shuffle feature

# Iterate through each feature. shuffling the feature n_repeat times.  calculate the change of score 
for f in range(F):
    scores = []
    for _ in range(n_repeats):
        Xp = X_test_scaled.copy()
        # shuffle feature f across all samples & timesteps
        flat = Xp[:,:,f].reshape(-1)
        np.random.shuffle(flat)
        Xp[:,:,f] = flat.reshape(n_test, T)
        scores.append(score(Xp, y_test_scaled))
    importances[f] = np.mean(scores) - base_score

# Map importances to feature names and sort
feat_imp = list(zip(features, importances))
feat_imp.sort(key=lambda x: x[1], reverse=True)
names, imps = zip(*feat_imp)

# Plot
plt.figure(figsize=(6,10))
plt.barh(names, imps)
plt.xlabel("MSE when feature is shuffled")
plt.title("Permutation Feature Importance (test set)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=300)
plt.show()
