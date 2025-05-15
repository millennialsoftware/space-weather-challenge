"""
Van_Transformer.py

Defines a pure transformer model for time‑series forecasting.  
Main components:
- Dense projection to an embedding dimension.
- Learnable positional embeddings.
- A stack of transformer encoder blocks (multi‑head attention + feed‑forward).
- Global average pooling and dense layers for the output head.
- Ray Tune helpers for simple hyperparameter search.

Utility helpers return the compiled model, standard training callbacks, and
dictionaries for default parameters and a Ray Tune search space.

Key Methods:
    get_model()               returns the compiled Keras model.
    get_callbacks()           early stopping and learning‑rate scheduler.
    get_hyperparameter_space() Ray Tune search grid.
    get_default_params()      baseline hyperparameter set.

Hyperparameter Tuning:
    Searchable items include learning_rate, batch_size, number of encoder
    layers, embedding size, attention heads, feed‑forward width, dropout rate,
    dense units, and activation functions.  See get_hyperparameter_space() for
    exact ranges.
"""

# Imports
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ray import tune  # For hyperparameter tuning

# Set GPU visibility (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1) -> None:
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training: bool = False) -> tf.Tensor:
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen: int, embed_dim: int) -> None:
        super().__init__()
        self.token_embeddings = Dense(embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        x = self.token_embeddings(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.position_embeddings(positions)
        return x + pos_embeddings

class VanillaTransformer:
    """
    A vanilla transformer model for time series forecasting.
    This version removes the CNN feature extraction and uses a stack of transformer encoder blocks.
    """
    def __init__(self, input_shape: tuple, horizon: int, learning_rate: float = 0.001,
                 batch_size: int = 8, epochs: int = 300, num_layers: int = 2, embed_dim: int = 64, 
                 num_heads: int = 4, ff_dim: int = 128, dropout_rate: float = 0.1, 
                 dense_units: int = 16, dense_activation: str = 'relu', 
                 final_dense_activation: str = 'linear') -> None:
        if len(input_shape) != 2:
            raise ValueError("input_shape should be a tuple of (timesteps, features)")
        
        self.input_shape = input_shape
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size  # <-- Added batch_size attribute
        self.epochs = epochs
        self.num_layers = num_layers
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
        """Defines the hyperparameter search space for Ray Tune (vanilla transformer version)."""
        return {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([8, 16, 32]),  # <-- Optionally include batch_size here
            "epochs": tune.choice([100, 200, 300, 500]),
            "num_layers": tune.choice([1, 2, 3, 4]),
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
        """Defines the default hyperparameters for the vanilla transformer."""
        return {
            "learning_rate": 0.001,
            "batch_size": 8,  # <-- Default batch_size included
            "epochs": 300,
            "num_layers": 2,
            "embed_dim": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "dropout_rate": 0.1,
            "dense_units": 16,
            "dense_activation": "relu",
            "final_dense_activation": "linear"
        }

    def _build_model(self) -> Model:
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Project the input features to the desired embedding dimension.
        x = Dense(self.embed_dim)(inputs)
        
        # Add learnable positional embeddings.
        x = PositionalEmbedding(maxlen=self.input_shape[0], embed_dim=self.embed_dim)(x)
        
        # Stack transformer encoder blocks.
        for _ in range(self.num_layers):
            x = TransformerBlock(embed_dim=self.embed_dim,
                                 num_heads=self.num_heads,
                                 ff_dim=self.ff_dim,
                                 rate=self.dropout_rate)(x)
        
        # Global pooling and dense layers for prediction.
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(self.dense_units, activation=self.dense_activation)(x)
        outputs = Dense(self.horizon, activation=self.final_dense_activation)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mae',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        return model

    def get_model(self) -> Model:
        """Returns the compiled vanilla transformer model."""
        return self.model

    def get_callbacks(self):
        """Returns training callbacks for stability."""
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1)
        ]
