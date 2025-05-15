"""
LSTM_Transformer.py

This script defines a Hybrid LSTM-Transformer model for time series forecasting using TensorFlow and Keras.
The model consists of:
- LSTM layers for sequential feature extraction.
- A Dense layer to project features to the desired embedding dimension.
- A learnable Positional Embedding layer to encode sequence position.
- A Transformer block for capturing long-term dependencies.
- Dense layers for final predictions.
- Hyperparameter tuning support using Ray Tune.

Dependencies:
    - os
    - tensorflow
    - ray.tune (for hyperparameter optimization)
"""

# Imports
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ray import tune  # For hyperparameter tuning

# Set GPU visibility (if applicable)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

class TransformerBlock(layers.Layer):
    """
    A Transformer block with Multi-Head Attention, feed-forward layers, 
    residual connections, and Layer Normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1) -> None:
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        """Performs forward propagation through the transformer block."""
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEmbedding(layers.Layer):
    """
    A learnable positional embedding layer to encode sequence position.
    """
    def __init__(self, maxlen: int, embed_dim: int) -> None:
        super().__init__()
        self.token_embeddings = Dense(embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = tf.shape(x)[1]
        x = self.token_embeddings(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.position_embeddings(positions)
        return x + pos_embeddings

class LSTM_Transformer:
    """
    A Hybrid LSTM-Transformer model for time series forecasting.
    
    This model replaces the convolutional layers with LSTM layers for sequential feature extraction.
    """
    def __init__(self, input_shape: tuple, horizon: int, learning_rate: float = 0.001, batch_size: int = 16,
                 epochs: int = 300, lstm_units1: int = 64, lstm_units2: int = 128, embed_dim: int = 64, 
                 num_heads: int = 4, ff_dim: int = 128, dropout_rate: float = 0.1,
                 dense_units: int = 16, dense_activation: str = 'relu',
                 final_dense_activation: str = 'relu') -> None:
        """
        Initializes the Hybrid LSTM-Transformer model with specified hyperparameters.
        """
        if len(input_shape) != 2:
            raise ValueError("input_shape should be a tuple of (timesteps, features)")

        self.input_shape = input_shape
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstm_units1 = lstm_units1
        self.lstm_units2 = lstm_units2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.final_dense_activation = final_dense_activation

        self.model = self._build_model()

    @classmethod
    def get_hyperparameter_space(cls):
        """Defines the hyperparameter search space for Ray Tune."""
        return {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([8, 16, 32]),
            "epochs": tune.choice([100, 200, 300, 500]),
            "lstm_units1": tune.choice([32, 64, 128]),
            "lstm_units2": tune.choice([64, 128, 256]),
            "embed_dim": tune.choice([32, 64, 128]),
            "num_heads": tune.choice([2, 4, 8]),
            "ff_dim": tune.choice([64, 128, 256]),
            "dropout_rate": tune.uniform(0.1, 0.5),
            "dense_units": tune.choice([8, 16, 32, 64]),
            "dense_activation": tune.choice(["relu", "tanh"]),
            "final_dense_activation": tune.choice(["linear", "relu", "tanh"])
        }
    
    @classmethod
    def get_default_params(cls):
        """Defines the default hyperparameters for Ray Tune Baseline."""
        return {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 300,
            "lstm_units1": 64,
            "lstm_units2": 128,
            "embed_dim": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout_rate": 0.1,
            "dense_units": 16,
            "dense_activation": "relu",
            "final_dense_activation": "relu"
        }

    def _build_model(self) -> Model:
        """
        Builds and compiles the Hybrid LSTM-Transformer model.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # LSTM-based feature extraction
        x = LSTM(self.lstm_units1, return_sequences=True, activation='relu')(inputs)
        x = LSTM(self.lstm_units2, return_sequences=True, activation='relu')(x)
        
        # Project LSTM outputs to embedding dimension
        x = Dense(self.embed_dim)(x)
        
        # Add positional embedding (helpful for Transformer layers)
        x = PositionalEmbedding(maxlen=x.shape[1], embed_dim=self.embed_dim)(x)
        
        # Transformer block to capture long-term dependencies
        x = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim, 
                               rate=self.dropout_rate)(x)
        
        # Global pooling and dense layers for final prediction
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.dense_units, activation=self.dense_activation)(x)
        outputs = Dense(self.horizon, activation=self.final_dense_activation)(x)
        
        # Build and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mae',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        return model

    def get_model(self) -> Model:
        """Returns the compiled Hybrid LSTM-Transformer model."""
        return self.model

    def get_callbacks(self):
        """Returns training callbacks for stability."""
        return [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1)
        ]
