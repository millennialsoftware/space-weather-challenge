"""Trains and evaluates a deep learning model using space weather data.

This module loads and preprocesses space weather data, applies feature scaling,
and trains a machine learning model using a 90/10 train-test split. It then evaluates
the model's performance using various metrics and saves the model and associated files
to the saved models folder.

The model can be configured with different feature sets, including:
  - Selected OMNI2 dataset features
  - GOES satellite data
  - Lagged time-series features
  - Missing data flags

Usage:
    from model_training import full_model_train

    full_model_train(
        selection,
        model, callbacks,
        omni2_fields=["Kp_index", "Dst_index_nT", "Flow_pressure"],
        include_goes=True,
        add_lags=True,
        add_flags=True,
        epochs=300,
        batch_size=8,
        transformer=None,
    )

"""
# Imports
import os
import tensorflow as tf
import logging
import shutil
import json
from preprocessing.load_data import load_data
from preprocessing.scaling import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from metrics import calculate_metrics
from typing import List
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import pickle
import uuid
from dotenv import load_dotenv

# Supress deep learning framework logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

def full_model_train(
    selection: str,
    model: Model,
    callbacks: List[Callback],
    omni2_fields: List[str] = [],
    include_goes: bool = False,
    add_lags: bool = False,
    add_flags: bool = False,
    epochs: int = 300,
    batch_size: int = 8,
    transformer: str = None,
) -> None:
    """Trains and evaluates a deep learning model using a 90/10 train-test split.

    This function loads and preprocesses the dataset based on the selected feature set,
    scales features and targets, and trains the provided model using the training data.
    After training, it evaluates the model on the validation set by calculating performance
    metrics and generates a submission file.

    Args:
        selection: (str): Model selection name, used for saving the model
        model (tf.keras.models.Model): A compiled deep learning model to be trained.
        callbacks (List[tf.keras.callbacks.Callback]): List of Keras callbacks to use during training.
        omni2_fields (List[str], optional): List of OMNI2 dataset features to include. Defaults to [].
        include_goes (bool, optional): Whether to include GOES satellite data. Defaults to False.
        add_lags (bool, optional): Whether to add lagged time-series features. Defaults to False.
        add_flags (bool, optional): Whether to include missing data flags. Defaults to False.
        epochs (int, optional): Number of training epochs. Defaults to 300.
        batch_size (int, optional): Batch size to use during training. Defaults to 8.
        transformer (str): Name of transformer model

    Returns:
        None.
    """

    # Load environment variables
    load_dotenv()

    # Set data path
    PATH = os.getenv("DATA_PATH")

    # Load training and validation sets
    X_all, y_all = load_data(path=PATH,include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)
    X_val, y_val = load_data(path=PATH,val=True,include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)

    # Scale data 

    # Set binary flags to be ignored from scaling 
    num_feats = X_all.shape[2]  # Total number of features

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
    X_scaled, _, X_val_scaled = scale_features(scaler_x, X_all, None, X_val, start_flag_index=start_flag_index)

    # Initialize the MinMaxScaler for target scaling
    scaler_y = MinMaxScaler()

    # Scale Targets
    y_scaled, _, y_val_scaled = scale_targets(scaler_y, y_all, None, y_val)

    # Fit the model on training data
    model.fit(
        X_scaled, y_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Test the model on test data
    y_pred_scaled = model.predict(X_val_scaled, verbose=0)

    # Undo transformations
    y_pred = undo_transformation(scaler_y, y_pred_scaled)
    y_test = undo_transformation(scaler_y, y_val_scaled)

    # Calculate rMSE, MAE, and sMAPE for predictions
    print("\n\n\n\033[32mTest Metrics: ")
    calculate_metrics(model, y_test, y_pred, save=False)

    # Save model to folder
    unique_id = str(uuid.uuid4())
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

    # Save to codabench folder
    with open(os.path.join(saved_directory, "parameters.json"), "w") as param_file:
        json.dump(parameters, param_file, indent=4)
    
     # Save model files
    model.save(os.path.join(saved_directory,"model.keras"))
    with open(os.path.join(saved_directory, "X_scaler.pkl"), "wb") as x_scaler_file:
        pickle.dump(scaler_x, x_scaler_file)
    with open(os.path.join(saved_directory, "y_scaler.pkl"), "wb") as y_scaler_file:
        pickle.dump(scaler_y, y_scaler_file)
    
    # If transformer model, save model to models folder
    if transformer is not None:
        destination_path = os.path.join(saved_directory, "transformer.py")
        source_path = "models/"+transformer +".py"
        shutil.copy2(source_path, destination_path)



    
