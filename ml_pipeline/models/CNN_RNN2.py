"""
CNN_RNN2.py

This script defines a multi-branch CNN-RNN model for time series forecasting using TensorFlow and Keras.
The model consists of:
- Three parallel convolutional branches with different kernel sizes to capture various temporal patterns.
- A merging layer to combine extracted features.
- LSTM and GRU layers for capturing sequential dependencies.
- Dense layers for final predictions.
- Hyperparameter tuning support using Ray Tune.

It also provides utility functions for retrieving the compiled model, training callbacks, 
and defining a hyperparameter search space for optimization.

Dependencies:
    - tensorflow
    - numpy (implicitly required by TensorFlow)
    - ray.tune (for hyperparameter optimization)

Methods:
    get_model(): Returns the compiled CNN-RNN model.
    get_callbacks(): Returns callbacks for early stopping and learning rate reduction.
    get_hyperparameter_space(): Returns a dictionary defining the hyperparameter search space for tuning.

Hyperparameter Tuning:
    This model supports hyperparameter tuning with Ray Tune. The available hyperparameters include:
        - learning_rate: Log-uniform search between 1e-4 and 1e-2.
        - batch_size: Choice between [8, 16, 32].
        - epochs: Choice between [100, 200, 300, 500].
        - conv_filters: Choice between [32, 64, 128].
        - kernel_sizes: Choice between [[3, 5, 7], [3, 5, 9], [3, 7, 11]].
        - lstm_units: Choice between [32, 64, 128].
        - lstm_dropout: Uniform search between 0.1 and 0.5.
        - gru_units: Choice between [16, 32, 64].
        - gru_dropout: Uniform search between 0.1 and 0.5.
        - dense_units: Choice between [8, 16, 32, 64].
        - dense_activation: Choice between ["relu", "tanh"].
"""

# Imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, concatenate
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ray import tune  # For hyperparameter tuning

class CNN_RNN2:
    """
    A multi-branch CNN-RNN hybrid model for time series forecasting.

    This model integrates three parallel convolutional branches with different kernel sizes 
    for feature extraction, followed by recurrent layers (LSTM and GRU) for sequential learning.

    Attributes:
        input_shape (tuple): Shape of the input time series data (timesteps, features).
        horizon (int): Number of future time steps to predict.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        kernel_sizes (list): List of kernel sizes for three convolutional branches.
        conv_filters (int): Number of filters in each convolutional branch.
        lstm_units (int): Number of LSTM units.
        lstm_dropout (float): Dropout rate for LSTM layer.
        gru_units (int): Number of GRU units.
        gru_dropout (float): Dropout rate for GRU layer.
        dense_units (int): Number of units in dense layers.
        dense_activation (str): Activation function for dense layers.
        model (Model): The compiled TensorFlow model.

    Methods:
        get_model(): Returns the compiled model.
        get_callbacks(): Returns training callbacks.
        get_hyperparameter_space(): Returns a dictionary of hyperparameter tuning space.
    """

    def __init__(self, input_shape: tuple, horizon: int, learning_rate: float = 0.001, batch_size: int = 8, epochs: int = 300,
                 kernel_sizes: list = [3, 5, 7], conv_filters: int = 64,
                 lstm_units: int = 64, lstm_dropout: float = 0.1,
                 gru_units: int = 32, gru_dropout: float = 0.1,
                 dense_units: int = 16, dense_activation: str = 'relu',
                 final_dense_activation: str = 'linear') -> None:
        """
        Initializes the CNN-RNN model with specified hyperparameters.

        Args:
            input_shape (tuple): Shape of the input time series data (timesteps, features).
            horizon (int): Number of future time steps to predict.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 8.
            epochs (int, optional): Number of training epochs. Defaults to 300.
            kernel_sizes (list, optional): Kernel sizes for the three convolutional branches. Defaults to [3, 5, 7].
            conv_filters (int, optional): Number of filters in each convolutional branch. Defaults to 64.
            lstm_units (int, optional): Number of units in the LSTM layer. Defaults to 64.
            lstm_dropout (float, optional): Dropout rate for the LSTM layer. Defaults to 0.1.
            gru_units (int, optional): Number of units in the GRU layer. Defaults to 32.
            gru_dropout (float, optional): Dropout rate for the GRU layer. Defaults to 0.1.
            dense_units (int, optional): Number of units in dense layers. Defaults to 16.
            dense_activation (str, optional): Activation function for dense layers. Defaults to 'relu'.
            final_dense_activation (str, optional): Activation function for the final dense layer. Defaults to 'linear'.
        """
        if len(input_shape) != 2:
            raise ValueError("input_shape should be a tuple of (timesteps, features)")

        self.input_shape = input_shape
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.kernel_sizes = kernel_sizes
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.gru_units = gru_units
        self.gru_dropout = gru_dropout
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.final_dense_activation = final_dense_activation
        
        self.model = self._build_model()

    @classmethod
    def get_hyperparameter_space(cls):
        """Defines the hyperparameter search space for Ray Tune"""
        return {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([8, 16, 32]),
            "epochs": tune.choice([100, 200, 300, 500]),
            "conv_filters": tune.choice([32, 64, 128]),
            "kernel_sizes": tune.choice([[3, 5, 7], [3, 5, 9], [3, 7, 11]]),
            "lstm_units": tune.choice([32, 64, 128]),
            "lstm_dropout": tune.uniform(0.1, 0.5),
            "gru_units": tune.choice([16, 32, 64]),
            "gru_dropout": tune.uniform(0.1, 0.5),
            "dense_units": tune.choice([8, 16, 32, 64]),
            "dense_activation": tune.choice(["relu", "tanh"]),
            "final_dense_activation": tune.choice(["linear"])
        }

    @classmethod
    def get_default_params(cls):
        """Defines the default hyperparameters for Ray Tune Baseline"""
        return {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 300,
            "kernel_sizes": [3, 5, 7],
            "conv_filters": 64,
            "lstm_units": 64,
            "lstm_dropout": 0.1,
            "gru_units": 32,
            "gru_dropout": 0.1,
            "dense_units": 16,
            "dense_activation": "relu",
            "final_dense_activation": "linear"
        }

    def _build_model(self) -> Model:
        """
        Builds and compiles the multi-branch CNN-RNN model.

        Returns:
            Model: The compiled Keras model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)

        # Create parallel CNN branches
        branches = []
        for kernel_size in self.kernel_sizes:
            x = Conv1D(self.conv_filters, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.1)(x)
            branches.append(x)

        # Merge branches
        merged = concatenate(branches, axis=-1)

        # Additional CNN layer post-merging
        merged = Conv1D(self.conv_filters * 2, kernel_size=5, activation='relu', strides=1, padding='same')(merged)
        merged = MaxPooling1D(pool_size=2)(merged)

        # RNN layers
        merged = LSTM(self.lstm_units, return_sequences=True, dropout=self.lstm_dropout)(merged)
        merged = GRU(self.gru_units, dropout=self.gru_dropout)(merged)

        # Dense layers
        x = Dense(self.dense_units, activation=self.dense_activation)(merged)
        outputs = Dense(self.horizon, activation=self.final_dense_activation)(x)

        # Build and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mae',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        return model

    def get_model(self) -> Model:
        """Returns the compiled CNN-RNN model."""
        return self.model

    def get_callbacks(self):
        """
        Returns a list of training callbacks.

        Returns:
            list: A list containing EarlyStopping and ReduceLROnPlateau callbacks.
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1)
        return [early_stop, reduce_lr]
