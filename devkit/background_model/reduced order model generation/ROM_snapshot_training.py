from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse

import numpy as np
import ray
from ray.tune import run, sample_from
from ray.rllib.utils import try_import_tf
from ray.tune.integration.keras import TuneReportCallback, TuneReportCheckpointCallback
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining

import os
#os.environ['GRPC_VERBOSITY']='DEBUG'
os.environ['TUNE_GLOBAL_CHECKPOINT_S']= '600' #Reduces frequency at which new checkpoints are saved

tf1, tf, tfv = try_import_tf()

def build_model(config):
    
    input_layer = tf.keras.layers.Input(shape=(36, 20, 24)) # 36,20,24 
    inputs_trans = tf.transpose(input_layer, [0, 2, 3, 1]) # Reshape to (20,24,36)
    
    ### Replace below with your network design ---------------------------------------------------------------------------------------------
    
    # Encoder layers
    encoder = tf.keras.layers.Conv2D(48, [8, 8], strides=(4, 4), padding="same", name="enc_conv1",activation=config["activation_fn"])(inputs_trans) 
    encoder = tf.keras.layers.BatchNormalization(axis=-1)(encoder)
    encoder = tf.keras.layers.Conv2D(64, [4, 4], strides=(2, 2), padding="same", name="enc_conv2",activation=config["activation_fn"])(encoder) 
    encoder = tf.keras.layers.BatchNormalization(axis=-1)(encoder)
    encoder = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), padding="same", name="enc_conv3",activation=config["activation_fn"])(encoder)
    encoder = tf.keras.layers.BatchNormalization(axis=-1)(encoder)
    encoder = tf.keras.layers.Flatten()(encoder)
    encoder= tf.keras.layers.Dense(
        288,
        name="enc_dense1",
        activation=config["activation_fn"])(encoder) #512
    encoder= tf.keras.layers.Dropout(config["dropout_ratio"])(encoder)
    latent= tf.keras.layers.Dense(
        10,
        name="enc_dense2",
        activation=None)(encoder) #ROM state: 100

    # Decoder layers
    decoder= tf.keras.layers.Dense(
        240,
        name="dec_dense1",
        activation=config["activation_fn"])(latent)
    decoder= tf.keras.layers.Dropout(config["dropout_ratio"])(decoder)
    decoder= tf.keras.layers.Dense(
        360,
        name="dec_dense2",
        activation=config["activation_fn"])(decoder)
    decoder = tf.keras.layers.Reshape((5,2,36))(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(64, [3, 3], strides=(1, 1), padding="valid", name="dec_conv1",activation=config["activation_fn"])(decoder)
    decoder = tf.keras.layers.BatchNormalization(axis=-1)(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(48, [3, 3], strides=(1, 1), padding="valid", name="dec_conv2",activation=config["activation_fn"])(decoder) 
    decoder = tf.keras.layers.BatchNormalization(axis=-1)(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(20, [8, 8], strides=(4, 4), padding="same", name="dec_conv3",activation=None)(decoder) 
    decoder = tf.transpose(decoder, [0, 1, 3, 2])

    ### Replace above with your network design ---------------------------------------------------------------------------------------------

    model = tf.keras.models.Model(input_layer, decoder)
    
    # Change Error Metric
    if config["error_met"] == "mae":
        model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"]),
        metrics=["mean_absolute_percentage_error","mean_squared_error","accuracy","mean_absolute_error"])
    elif config["error_met"] == "mse":
        model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"]),
        metrics=["mean_absolute_percentage_error","mean_squared_error","accuracy","mean_absolute_error"])
    else:
        model.compile(
        loss="mean_absolute_percentage_error",
        optimizer=tf.keras.optimizers.Adam(
            lr=config["lr"]),
        metrics=["mean_absolute_percentage_error","mean_squared_error","accuracy","mean_absolute_error"])
    
    return model

def train_agent(config, x_train=None, x_val=None):
    batch_size = config["batch"]
    epochs = 100000
    model = build_model(config)

    model.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_val, x_val),
        callbacks=[TuneReportCheckpointCallback(metrics={
            "MAPE": "mean_absolute_percentage_error", "MSE":"mean_squared_error", "MAE":"mean_absolute_error",
            "val_MAPE": "val_mean_absolute_percentage_error", "val_MSE":"val_mean_squared_error","val_MAE":"val_mean_absolute_error"
        }, frequency=49, filename="model", on="epoch_end")])


if __name__ == "__main__":
    # ensure temp_dir is correct----------------------------------------------------------------------------------------------------------
    ray.init(num_cpus=10, num_gpus=0, _temp_dir="temp_dir")
#-----------------------------------------------------------------------------------------------------------------------------------------
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=100, resample_probability=0.25, #120
        quantile_fraction=0.3,

        # Specify hyperparams below------------------------------------------------------------------------------------------------------
        hyperparam_mutations={  
            "lr": lambda: random.uniform(1e-6, 1e-2),
            "dropout_ratio": lambda: random.uniform(0, 0.3),
            "batch": [32, 64, 80, 96, 128, 160, 196, 240, 256, 384, 512],
            "activation_fn": ["relu", "elu"],
            "error_met": ["mae", "mse","mape"]
        })
        #--------------------------------------------------------------------------------------------------------------------------------
    # # Global Variables
    # MAX_EVALS = 1000
    # EPOCHS = 300
    # batch_size = 256 
    DATA_PATH = 'data/'
#     MODEL_PATH = '../models/'
    

    #Load_DATA
    
    x_train_obj = np.load(DATA_PATH+"xtrain.npy")
    x_val_obj = np.load(DATA_PATH+"xtest.npy")

    analysis = tune.run(
        tune.with_parameters(train_agent, x_train=x_train_obj, x_val=x_val_obj),
        name="full",
        # ensure local_dir is correct-------------------------------------------------------------------------------------------------------
        local_dir="runs",
        #-----------------------------------------------------------------------------------------------------------------------------------
#         resume=True,
        scheduler=pbt,
        metric="val_MAPE",
        mode="min",
        checkpoint_freq=100,
        max_failures=50,
        stop={
            "training_iteration": 100000
        },
        num_samples=4,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        # Specify hyperparams below (for network init)--------------------------------------------------------------------------------------
        config={
            "lr": sample_from(lambda spec: random.uniform(0.0001, 0.05)),
            "batch": sample_from(lambda spec: random.randint(32, 512)),
            "activation_fn": sample_from(lambda spec: random.choice(["relu", "elu"])),
            "error_met": sample_from(lambda spec: random.choice(["mae", "mse","mape"])),
            "dropout_ratio": sample_from(lambda spec: random.uniform(0, 0.1)),
        })
        #-----------------------------------------------------------------------------------------------------------------------------------
    print("Best hyperparameters found were: ", analysis.best_config)
    best_trial = analysis.get_best_trial("val_MAPE", "min", "last")
    print("Best Checkpoint is at: ", best_trial.checkpoint.value)




