"""Tune Model Module.

This module performs hyperparameter tuning for deep learning models designed to forecast space weather data.
It loads and preprocesses the dataset, scales features and targets, and splits the data into training,
validation, and testing sets. Hyperparameter tuning is conducted using Ray Tune with an ASHA scheduler
and HyperOpt search algorithm. After tuning, the best hyperparameter configuration is saved to a JSON
file ('models/best_params.json') and then used to train a final model on the full dataset.

Available Models:
    - CNN_RNN
    - CNN_RNN2
    - CNN_Transformer
    - Van_Transformer
    - LSTM Transformer
    - CNN_RNN_transformer
    - CNN_RNN_Transfromer_Att
    - CNN_RNN_TransfromerEncoder

Notes:
    - Transformer Parameter: For transformer-based models (e.g., CNN_Transformer, Van_Transformer, LSTM Transformer,
      and the CNN_RNN_transformer variants), the module assigns a string (the model's name) to the parameter
      'transformer'. For non-transformer-based models, the 'transformer' parameter is set to None.
      When saved to JSON, Python’s None is automatically converted to null.
    - Checkpointing: If a previous tuning checkpoint is detected for a model, the module attempts to resume tuning
      from that checkpoint rather than starting from scratch.
    - Email Updates: The module can optionally send email updates upon finishing tuning a model if enabled by the user.
    - Data Processing: The module scales features using StandardScaler and targets using MinMaxScaler. It adjusts scaling
      for additional binary flag features (if applicable).
"""

# Imports
from preprocessing.load_data import load_data
from preprocessing.scaling import scale_features, scale_targets, undo_transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import json
from full_train import full_model_train
from email_services import send_email
from typing import Dict, Any, Optional
import shutil
import os
from dotenv import load_dotenv

# Model imports
from models.CNN_RNN import CNN_RNN
from models.CNN_RNN2 import CNN_RNN2
from models.CNN_Transformer import CNN_Transformer
from models.Van_Transformer import VanillaTransformer
from models.LSTM_Tranformer import LSTM_Transformer
from models.CNN_RNN_transformer import cnn_lstm_gru_transformer
from models.CNN_RNN_Transformer_Att import cnn_lstm_gru_transformer_selfAtt
from models.CNN_RNN_TransfromerEncoder import cnn_lstm_gru_transformerEncoder

# Ray Tune Imports
import ray 
from ray import tune
from ray.train.tensorflow.keras import ReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
ray.init(logging_level="ERROR")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def train_model_tune(config: Dict[str, Any], model_class: type, data: Optional[Dict[str, Any]] = None) -> None:
    """Train a model with Ray Tune and report the best validation loss."""
    # Retrieve preprocessed data
    X_train_scaled = data["X_train_scaled"]
    y_train_scaled = data["y_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_test_scaled = data["y_test_scaled"]

    # Instantiate model using hyperparameters
    model_obj = model_class(
        input_shape=data["input_shape"],
        horizon=data["horizon"],
        **config
    )
    model = model_obj.get_model()

    # Get callbacks and add the Tune checkpoint callback
    callbacks = model_obj.get_callbacks()
    callbacks.append(ReportCheckpointCallback())

    # Train the model and store history
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=0
    )

    # Extract the best validation loss and report to Ray Tune
    best_val_loss = min(history.history["val_loss"])
    tune.report({"val_loss": best_val_loss})

class HyperparameterTuner:
    """
    A class for conducting hyperparameter tuning on deep learning models for space weather forecasting.

    This class leverages Ray Tune to explore the hyperparameter search space for a given model. It integrates
    an ASHA scheduler for early stopping of underperforming trials and the HyperOpt search algorithm for efficient
    hyperparameter optimization. The tuner supports checkpoint resumption, email notifications upon tuning completion,
    and updates a central JSON file with the best hyperparameter configurations..

    Attributes:
        num_trials (int): Number of hyperparameter tuning trials.
        data (Dict[str, Any]): Preprocessed input and target data along with model configuration details.
        best_param_dict (Dict[str, Any]): Dictionary containing the best-known hyperparameters for each model.
        email_updates (bool): Flag to enable or disable email notifications.
        resources (Dict[str, int]): Resource allocation for each Ray Tune trial.
        stop_config (Dict[str, int]): A configuration dict that defines stopping criteria for the training iterations.

    Methods:
        tune_model(model_name: str, ModelClass: type) -> Dict[str, Any]:
            Sets up and runs hyperparameter tuning for the specified model. This method:
              - Retrieves the hyperparameter space from the given model class.
              - Configures a checkpoint directory for resuming tuning sessions if available.
              - Wraps the training function with model and data parameters.
              - Initializes Ray Tune with an ASHA scheduler and HyperOpt search algorithm.
              - Runs the tuning trials and identifies the best hyperparameter configuration based on 
                validation loss.
              - Optionally sends an email update with the tuning results.
              - Updates and saves the best parameters to a JSON file.
              - Removes tuning checkpoint directories after successful tuning.

    Usage:
        tuner = HyperparameterTuner(num_trials, data, best_param_dict, email_updates)
        best_config = tuner.tune_model(model_name, ModelClass)
    """
    def __init__(self, num_trials: int, data: Dict[str, Any], best_param_dict: Dict[str, Any],
                 email_updates: bool, resources: Dict[str, int] = {"cpu": 48, "gpu": 1}):
        self.num_trials = num_trials
        self.data = data
        self.best_param_dict = best_param_dict
        self.email_updates = email_updates
        self.resources = resources
        self.stop_config = {"training_iteration": 300}

    def tune_model(self, model_name: str, ModelClass: type) -> Dict[str, Any]:
        """Set up and run Ray Tune for a given model."""
        param_space = ModelClass.get_hyperparameter_space()
         # Define the checkpoint directory specific to the model.
        
        # Define a relative directory for checkpoints
        relative_checkpoint_dir = "raytune"

        # Convert to an absolute path
        abs_checkpoint_dir = os.path.abspath(relative_checkpoint_dir)

        # Prepend the "file://" scheme to create a proper URI
        storage_path = f"file://{abs_checkpoint_dir}"

        # if model checkpoints exists set resume to True 
        resume = False
        model_checkpoint_dir = os.path.join(abs_checkpoint_dir, model_name)
        if os.path.exists(model_checkpoint_dir):
            resume = True

        # Wrap the training function with model and data parameters
        trainable = tune.with_resources(
            tune.with_parameters(train_model_tune, model_class=ModelClass, data=self.data),
            self.resources
        )

        # Initialize ASHAScheduler to monitor 'val_loss' and early stop underperforming trials
        asha_scheduler = ASHAScheduler(
            time_attr='epoch',
            metric='val_loss',              
            mode='min',
            max_t=300,                     
            grace_period=15,
            reduction_factor=3,             
            brackets=1                   
        )
    
        # Initialize HyperOptSearch to minimize 'val_loss', using baseline hyperparameters from best_param_dict if available.
        hyperopt_search = HyperOptSearch(
           metric="val_loss",
           mode="min",
           points_to_evaluate=[self.best_param_dict.get(model_name)]
                if self.best_param_dict.get(model_name)
                else [ModelClass.get_default_params()]
        )
    
        # Initialize the Ray Tune Tuner with the training function, hyperparameter search algorithm, and run configuration.
        tuner_config = tune.Tuner(
            trainable,  
            tune_config=tune.TuneConfig(
                num_samples=self.num_trials,
                scheduler=asha_scheduler,
                search_alg=hyperopt_search,
                max_concurrent_trials=1
            ),
            run_config=tune.RunConfig(
                storage_path=storage_path,  # This is your custom path.
                name=model_name,             
                stop={"training_iteration": 300},
                verbose=1
            ),
            param_space=param_space,
        )

        # If the checkpoint dir already exists, restore from the previous tuning session.
        if resume:
            try:
                tuner = tune.Tuner.restore(model_checkpoint_dir, trainable=trainable)
                print(f"Resuming tuning for {model_name} from checkpoint in {model_checkpoint_dir}.")
            except Exception as e:
                print(f"Could not restore from checkpoint in {model_checkpoint_dir}. Starting new tuning session. Error: {e}")
                tuner = tuner_config
        else:
            tuner = tuner_config


        # Run hyperparameter tuning
        results = tuner.fit()
        best_result = results.get_best_result(metric="val_loss", mode="min")
        best_config = best_result.config
        print(f"Best hyperparameters found for {model_name}:", best_config)

        # Send email update if enabled
        if self.email_updates:
            email_content = f"""
            <html>
            <body>
                <h2 style="color:green; text-align: center;">Ray Tune Update</h2>
                <p>Model <b>{model_name}</b> finished tuning.</p>
                <p><b>Best Parameters: </b> {best_config}</p>
                <hr>
            </body>
            </html>
            """
            send_email(subject=f"Ray Tune Update for {model_name}", content=email_content)

        # Update and save best parameters
        self.best_param_dict[model_name] = best_config
        with open("models/best_params.json", "w") as f:
            json.dump(self.best_param_dict, f, indent=4)
        
        # Delete the checkpoint directory once tuning is complete.
        if os.path.exists(model_checkpoint_dir):
            try:
                shutil.rmtree(model_checkpoint_dir)
                print(f"Deleted checkpoint directory: {model_checkpoint_dir}")
            except Exception as e:
                print(f"Failed to delete checkpoint directory {model_checkpoint_dir}: {e}")

        return best_config


if __name__ == "__main__":
    
    # Number of trials to perform
    num_trials = 1

    # Load or create best parameters dictionary
    try:
        with open("models/best_params.json", "r") as file:
            best_param_dict = json.load(file)
    except FileNotFoundError:
        best_param_dict = {}

    # Dictionary mapping numbers to (string representation, class reference)
    models = {
        1: ("CNN_RNN", CNN_RNN),
        2: ("CNN_RNN2", CNN_RNN2),
        3: ("CNN_Transformer", CNN_Transformer),
        4: ("Van_Transformer", VanillaTransformer),
        5: ("LSTM Transformer", LSTM_Transformer),
        6: ("CNN_RNN_transformer", cnn_lstm_gru_transformer),
        7: ("CNN_RNN_Transfromer_Att", cnn_lstm_gru_transformer_selfAtt),
        8: ("CNN_RNN_TransfromerEncoder", cnn_lstm_gru_transformerEncoder)
    }

    # Display available models
    print("\nAvailable Models:")
    for key, (name, _) in models.items():
        print(f"{key}: {name}")

    # Get user selection for tuning a single model or all models
    selection = None
    while selection not in list(models.keys()) + [999]:
        try:
            selection = int(input("Select a model to tune or 999 to tune all: "))
        except ValueError:
            print("Invalid input! Please enter a number.")

    # Ask user if they want email updates
    email_choice = input("Do you want email updates? (Y/N): ")
    email_updates = email_choice.lower() == 'y'

    tune_all = (selection == 999)
    if not tune_all:
        ModelClass = models[selection][1]
    
    if selection in [3, 4, 5, 6, 7, 8]:
        transformer = models[selection][0]
    else:
        transformer = None
      
    # Best features found from feature exhaustion
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
    
    # Set number of files
    num_files = 10

    # Data processing params
    include_goes = False
    add_lags = False
    add_flags = True

    # Validation set size
    num_val_files = num_files // 10

    # Load environment variables
    load_dotenv()

    PATH = os.getenv("DATA_PATH")

    # Load training and validation sets
    X_all, y_all = load_data(path=PATH, num_files=num_files, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)
    X_val, y_val = load_data(path=PATH, num_files=num_val_files, val=True, include_goes=include_goes, add_lags=add_lags, omni2_fields=omni2_fields, add_flags=add_flags)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, shuffle=True
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

    data = {
        "X_train_scaled": X_train_scaled,
        "y_train_scaled": y_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_test_scaled": y_test_scaled,
        "input_shape": (X_train_scaled.shape[1], X_train_scaled.shape[2]),
        "horizon": y_train.shape[1]
    }

    # Create an instance of the tuner
    tuner_instance = HyperparameterTuner(num_trials=num_trials, data=data,
                                         best_param_dict=best_param_dict,
                                         email_updates=email_updates)
    
    # tune just one model of choice. Save tuned model to submission folder 
    if not tune_all:
        model_name = models[selection][0]
        print("Tuning ",model_name)
        best_config = tuner_instance.tune_model(model_name, ModelClass)
        # Initialize and train the final model
        model_obj = ModelClass(input_shape=data["input_shape"], horizon=data["horizon"], **best_config)
        final_model = model_obj.get_model()
        full_model_train(selection=model_name, model=final_model, callbacks=model_obj.get_callbacks(), omni2_fields=omni2_fields,
                         include_goes=include_goes, add_lags=add_lags, add_flags=add_flags,
                         epochs=best_config["epochs"], batch_size=best_config["batch_size"],transformer=transformer)
   
    # Tune all models one after the other. saves final tuned model to a submission folder
    else:
        for key, (model_name, ModelClass) in models.items():
            
            if key in [3, 4, 5, 6, 7, 8]:
                transformer = model_name
            else:
                transformer = None
                
            print("Tuning ",model_name)
            best_config = tuner_instance.tune_model(model_name, ModelClass)
            model_obj = ModelClass(input_shape=data["input_shape"], horizon=data["horizon"], **best_config)
            final_model = model_obj.get_model()
            full_model_train(selection=model_name, model=final_model, callbacks=model_obj.get_callbacks(), omni2_fields=omni2_fields,
                         include_goes=include_goes, add_lags=add_lags, add_flags=add_flags,
                         epochs=best_config["epochs"], batch_size=best_config["batch_size"],transformer=transformer)