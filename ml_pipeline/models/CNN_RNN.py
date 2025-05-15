"""
CNN_RNN.py

This script defines a CNN-RNN hybrid model using TensorFlow and Keras for time series forecasting. 
The model consists of:
- 1D Convolutional layers for feature extraction.
- LSTM and GRU layers for capturing sequential dependencies.
- Dense layers for final predictions.
- Hyperparameter tuning support using Ray Tune.

It also provides utility functions for retrieving the compiled model, training callbacks, 
and defining a hyperparameter search space for optimization.

Environment Variables:
    TF_CPP_MIN_LOG_LEVEL (str): Controls TensorFlow logging verbosity.
    TF_GPU_ALLOCATOR (str): Defines the GPU memory allocation strategy.

Dependencies:
    - os
    - tensorflow
    - numpy (implicitly required by TensorFlow)
    - ray.tune (for hyperparameter optimization)

Methods:
    get_model(): Returns the compiled CNN-RNN model.
    get_callbacks(monitor_validation=True): Returns callbacks for early stopping and learning rate reduction.
    get_hyperparameter_space(): Returns a dictionary defining the hyperparameter search space for tuning.

Hyperparameter Tuning:
    This model supports hyperparameter tuning with Ray Tune. The available hyperparameters include:
        - learning_rate: Log-uniform search between 1e-4 and 1e-2.
        - batch_size: Choice between [8, 16, 32].
        - epochs: Fixed at 300 but can be modified.
        - conv_filters1: Choice between [32, 64, 128].
        - conv_filters2: Choice between [64, 128, 256].
        - kernel_size: Choice between [3, 5, 7].
        - lstm_units: Choice between [32, 64, 128].
        - lstm_dropout: Uniform search between 0.1 and 0.5.
        - gru_units: Choice between [16, 32, 64].
        - gru_dropout: Uniform search between 0.1 and 0.5.
        - dense_units: Choice between [8, 16, 32, 64].
        - dense_activation: Choice between ["relu", "tanh"].
"""

# Imports
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, BatchNormalization,SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ray import tune  # Needed for hyperparameter tuning
from tensorflow.keras.losses import Huber
# Configure TensorFlow logging and GPU settings
gpus = tf.config.list_physical_devices('GPU')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TF log verbosity
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.get_logger().setLevel('ERROR')

class CNN_RNN:
    """
    A CNN-RNN hybrid model for time series forecasting.

    This model integrates convolutional layers for feature extraction with 
    recurrent layers (LSTM and GRU) for sequential learning. It is designed 
    for predicting future time steps in a given time series dataset.

    Attributes:
        input_shape (tuple): Shape of the input time series data (timesteps, features).
        horizon (int): Number of future time steps to predict.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        conv_filters1 (int): Number of filters in the first Conv1D layer.
        conv_filters2 (int): Number of filters in the second Conv1D layer.
        kernel_size (int): Kernel size for Conv1D layers.
        lstm_units (int): Number of units in the LSTM layer.
        lstm_dropout (float): Dropout rate for the LSTM layer.
        gru_units (int): Number of units in the GRU layer.
        gru_dropout (float): Dropout rate for the GRU layer.
        dense_units (int): Number of units in the first dense layer.
        dense_activation (str): Activation function for the first dense layer.
        model (Sequential): The compiled TensorFlow model.

    Methods:
        get_model(): Returns the compiled model.
        get_callbacks(monitor_validation=True): Returns training callbacks.
    """

    def __init__(self, input_shape: tuple, horizon: int, learning_rate: float = 0.001, batch_size: int = 8,
                 epochs: int = 300, conv_filters1: int = 64, conv_filters2: int = 128, kernel_size: int = 5,
                 lstm_units: int = 64, lstm_dropout: float = 0.1, gru_units: int = 32, gru_dropout: float = 0.1,
                 dense_units: int = 16, dense_activation: str = 'relu',) -> None:
        """
        Initializes the CNN-RNN model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of the input time series data (timesteps, features).
            horizon (int): Number of future time steps to predict.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 8.
            epochs (int, optional): Number of training epochs. Defaults to 300.
            conv_filters1 (int, optional): Number of filters in the first Conv1D layer. Defaults to 64.
            conv_filters2 (int, optional): Number of filters in the second Conv1D layer. Defaults to 128.
            kernel_size (int, optional): Kernel size for Conv1D layers. Defaults to 5.
            lstm_units (int, optional): Number of units in the LSTM layer. Defaults to 64.
            lstm_dropout (float, optional): Dropout rate for the LSTM layer. Defaults to 0.1.
            gru_units (int, optional): Number of units in the GRU layer. Defaults to 32.
            gru_dropout (float, optional): Dropout rate for the GRU layer. Defaults to 0.1.
            dense_units (int, optional): Number of units in the first dense layer. Defaults to 16.
            dense_activation (str, optional): Activation function for the first dense layer. Defaults to 'relu'.
            final_dense_activation (str, optional): Activation function for the final dense layer. Defaults to 'linear'.
        """
        self.input_shape = input_shape
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.conv_filters1 = conv_filters1
        self.conv_filters2 = conv_filters2
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.gru_units = gru_units
        self.gru_dropout = gru_dropout
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        
        self.model = self._build_model()
    
    @classmethod
    def get_hyperparameter_space(cls):
        """Defines the hyperparameter search space for Ray Tune"""
        return {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([8, 16, 32]),
            "epochs": 300,
            "conv_filters1": tune.choice([32, 64, 128]),
            "conv_filters2": tune.choice([64, 128, 256]),
            "kernel_size": tune.choice([3, 5, 7]),
            "lstm_units": tune.choice([32, 64, 128]),
            "lstm_dropout": tune.uniform(0.1, 0.5),
            "gru_units": tune.choice([16, 32, 64]),
            "gru_dropout": tune.uniform(0.1, 0.5),
            "dense_units": tune.choice([8, 16, 32, 64]),
            "dense_activation": tune.choice(["relu", "tanh"]),
        }
    
    @classmethod
    def get_default_params(cls):
        """Defines default params space for Ray Tune Baseline"""
        return {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 110,
            "conv_filters1": 64,
            "conv_filters2": 128,
            "kernel_size": 5,
            "lstm_units": 64,
            "lstm_dropout": 0.1,
            "gru_units": 32,
            "gru_dropout": 0.1,
            "dense_units": 16,
            "dense_activation": 'relu',
        }

    def _build_model(self) -> Sequential:
        """
        Builds and compiles the CNN-RNN model.

        Returns:
            Sequential: The compiled Keras model.
        """
        model = Sequential()

        # 1D Convolutional Layers
        model.add(Conv1D(filters=self.conv_filters1, kernel_size=self.kernel_size, strides=1,
                         activation='relu', input_shape=self.input_shape))


        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=self.conv_filters2, kernel_size=self.kernel_size, strides=1, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        # LSTM & GRU Layers
        model.add(LSTM(self.lstm_units, return_sequences=True, dropout=self.lstm_dropout))
        model.add(GRU(self.gru_units, dropout=self.gru_dropout))

        # Dense Layers
        model.add(Dense(self.dense_units, activation=self.dense_activation))
        model.add(Dense(self.horizon, activation='linear'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=Huber(delta=0.1),
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        return model

    def get_model(self) -> Sequential:
        """
        Returns the compiled CNN-RNN model.

        Returns:
            Sequential: The compiled Keras model.
        """
        return self.model

    def get_callbacks(self):
        """
        Returns a list of Keras callbacks for training, including early stopping and learning rate reduction.

        Args:
            monitor_validation (bool, optional): If True, monitors `val_loss`. Otherwise, monitors `loss`. Defaults to True.

        Returns:
            list: A list containing EarlyStopping and ReduceLROnPlateau callbacks.
        """
        monitor_metric = 'val_loss'

        early_stop = EarlyStopping(monitor=monitor_metric, patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.1, patience=15, min_lr=1e-7, verbose=1)

        return [early_stop, reduce_lr]