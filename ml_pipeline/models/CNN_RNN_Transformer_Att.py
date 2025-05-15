"""
CNN_RNN_Transformer_Att.py

This script defines a hybrid neural network for time‑series forecasting that
combines:
- 1‑D convolution and max‑pooling layers for local feature extraction.
- A small transformer block for global self‑attention.
- LSTM and GRU layers for sequential context.
- A custom self‑attention layer over the RNN outputs.
- Dense layers for final predictions.
- Ray Tune hooks for simple hyperparameter search.

Utility functions are provided for returning the compiled model, standard
training callbacks, and dictionaries describing default parameters and a Ray
Tune search space.

Key Methods:
    get_model()               – returns the compiled Keras model.
    get_callbacks()           – early‑stopping and learning‑rate scheduler.
    get_hyperparameter_space() – Ray Tune search grid.
    get_default_params()      – baseline hyperparameter set.

Hyperparameter Tuning:
    Exposes common settings such as learning_rate, batch_size, filter sizes,
    kernel_size, RNN units, dropout rates, dense units, and transformer heads.
    See get_hyperparameter_space() for exact ranges.
"""

# Imports 
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ray import tune 

# Configure TensorFlow logging and GPU settings
gpus = tf.config.list_physical_devices('GPU')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TF log verbosity
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.get_logger().setLevel('ERROR')

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="att_weight"
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
            name="att_bias"
        )
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute the attention scores
        # score shape: (batch_size, timesteps, 1)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # Normalize the scores to probabilities over the time dimension
        attention_weights = tf.nn.softmax(score, axis=1)
        # Multiply the inputs by the attention weights and sum over time
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector
        
class TransformerBlock(tf.keras.layers.Layer):
    """
    Minimal custom Transformer block with unique layer names.
    """
    def __init__(self, 
             num_heads=2, 
             key_dim=32, 
             ff_dim=32, 
             dropout_rate=0.1, 
             block_name="transformer_block",
             **kwargs):
        kwargs.pop("name", None)
        super(TransformerBlock, self).__init__(name=block_name, **kwargs)

        # Remember hyperparameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Sub-layers with explicit unique names
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim,
            name=f"{self.name}_mha"
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_ln1"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_ln2"
        )
        self.dropout1 = tf.keras.layers.Dropout(
            self.dropout_rate,
            name=f"{self.name}_dropout1"
        )
        self.dropout2 = tf.keras.layers.Dropout(
            self.dropout_rate,
            name=f"{self.name}_dropout2"
        )

    def build(self, input_shape):
        """
        Build the feed-forward network (FFN) after we know the input shape.
        """
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.ff_dim,
                    activation='relu',
                    name=f"{self.name}_ffn_dense1"
                ),
                tf.keras.layers.Dense(
                    input_shape[-1],
                    name=f"{self.name}_ffn_dense2"
                ),
            ],
            name=f"{self.name}_ffn"
        )
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass through the multi-head attention, residual connections, and FFN.
        """
        # Multi-head self-attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class cnn_lstm_gru_transformer_selfAtt:
    """
    A CNN-RNN hybrid model (CNN + Transformer + LSTM + GRU) for time series forecasting.
    Includes explicit unique layer names to avoid naming collisions.
    """

    def __init__(
        self, 
        input_shape: tuple, 
        horizon: int, 
        learning_rate: float = 0.001, 
        batch_size: int = 8, 
        epochs: int = 300,
        conv_filters1: int = 64, 
        conv_filters2: int = 128, 
        kernel_size: int = 5,
        lstm_units: int = 64, 
        lstm_dropout: float = 0.1,
        gru_units: int = 32, 
        gru_dropout: float = 0.1,
        dense_units: int = 16, 
        dense_activation: str = 'relu',
        final_dense_activation: str = 'relu',
        transformer_num_heads: int = 2, 
        transformer_key_dim: int = 32, 
        transformer_ff_dim: int = 32, 
        transformer_dropout_rate: float = 0.1
    ) -> None:

        self.input_shape = input_shape
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # CNN Hyperparams
        self.conv_filters1 = conv_filters1
        self.conv_filters2 = conv_filters2
        self.kernel_size = kernel_size

        # LSTM/GRU Hyperparams
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.gru_units = gru_units
        self.gru_dropout = gru_dropout

        # Dense Hyperparams
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.final_dense_activation = final_dense_activation

        # Transformer Hyperparams
        self.transformer_num_heads = transformer_num_heads
        self.transformer_key_dim = transformer_key_dim
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout_rate = transformer_dropout_rate
        
        self.model = self._build_model()
    
    @classmethod
    def get_hyperparameter_space(cls):
        """Defines the hyperparameter search space for Ray Tune."""
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
            "final_dense_activation": tune.choice(["linear", "relu", "tanh"]),
            "transformer_num_heads": tune.choice([1, 2, 4]),
            "transformer_key_dim": tune.choice([16, 32, 64]),
            "transformer_ff_dim": tune.choice([16, 32, 64]),
            "transformer_dropout_rate": tune.uniform(0.1, 0.5)
        }
    
    @classmethod
    def get_default_params(cls):
        """Defines default hyperparameters for Ray Tune Baseline."""
        return {
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 300,
            "conv_filters1": 64,
            "conv_filters2": 128,
            "kernel_size": 5,
            "lstm_units": 64,
            "lstm_dropout": 0.1,
            "gru_units": 32,
            "gru_dropout": 0.1,
            "dense_units": 16,
            "dense_activation": 'relu',
            "final_dense_activation": 'linear',
            "transformer_num_heads": 2,
            "transformer_key_dim": 32,
            "transformer_ff_dim": 32,
            "transformer_dropout_rate": 0.1
        }

    def _build_model(self) -> Sequential:
        """
        Builds and compiles the CNN + Transformer + LSTM + GRU model.
        """
        model = Sequential(name="cnn_lstm_gru_transformer_model")
        

      
         # 1) Convolutional Layers
        model.add(
            Conv1D(
                filters=self.conv_filters1,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
                input_shape=self.input_shape,
                name="conv1d_layer1"
            )
        )
        model.add(
            MaxPooling1D(
                pool_size=2,
                name="maxpool1d_layer1"
            )
        )

        
        model.add(
            Conv1D(
                filters=self.conv_filters2,
                kernel_size=self.kernel_size,
                strides=1,
                activation='relu',
                name="conv1d_layer2"
            )
        )
        model.add(
            MaxPooling1D(
                pool_size=2,
                name="maxpool1d_layer2"
            )
        )

        #Transformer block add 
        model.add(
            TransformerBlock(
                num_heads=self.transformer_num_heads,
                key_dim=self.transformer_key_dim,
                ff_dim=self.transformer_ff_dim,
                dropout_rate=self.transformer_dropout_rate,
                block_name="transformer_block"
            )
        )

        model.add(
            LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.lstm_dropout,
                name="lstm_layer"
            )
        )
       
             
        # 4) GRU Layer with full sequence output
        model.add(
            tf.keras.layers.GRU(
                self.gru_units,
                return_sequences=True,  # Needed for attention over time steps
                dropout=self.gru_dropout,
                name="gru_layer"
            )
        )
        
        # 5) Self-Attention Layer on RNN outputs
        model.add(
            SelfAttention(name="self_attention_layer")
        )
        
        # 4) Dense Layers
        model.add(
            Dense(
                self.dense_units, 
                activation=self.dense_activation,
                name="dense_hidden"
            )
        )
        model.add(
            Dense(
                self.horizon, 
                activation=self.final_dense_activation,
                name="dense_output"
            )
        )

               # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mae',
                      metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        return model

    def get_model(self) -> Sequential:
        """
        Returns the compiled CNN-RNN model (CNN + Transformer + LSTM + GRU).
        """
        return self.model

    def get_callbacks(self):
        """Returns training callbacks for stability."""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=15,
                min_lr=1e-7,
                verbose=1
            )
        ]
