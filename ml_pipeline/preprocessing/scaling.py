"""
Feature Scaling and Transformation Utilities.

This module provides functions for scaling and transforming time-series 
features and target variables in machine learning models. It includes 
methods for scaling numerical inputs while preserving binary flags, 
scaling target variables, and reversing transformations.

Functions:
    - scale_features: Scales numerical features while keeping binary flags unchanged.
    - scale_targets: Scales target variables for training, testing, and validation.
    - undo_transformation: Reverts the scaling and log transformation of target values.
"""

# Imports
import numpy as np
import numpy as np
from sklearn.base import TransformerMixin
from typing import Optional, Tuple

def scale_features(
    scaler: TransformerMixin,
    train: np.ndarray,
    test: Optional[np.ndarray] = None,
    val: Optional[np.ndarray] = None,
    start_flag_index: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Scales numerical features while keeping binary flags unchanged.

    This function scales only the numerical features in 3D time-series datasets, 
    preserving any binary flags present in the latter part of the feature set.

    Args:
        scaler (TransformerMixin): A scikit-learn scaler instance (e.g., MinMaxScaler, StandardScaler).
        train (np.ndarray): Training dataset with shape (num_samples, seq_len, num_features).
        test (Optional[np.ndarray], optional): Test dataset with the same shape as `train`. Defaults to None.
        val (Optional[np.ndarray], optional): Validation dataset with the same shape as `train`. Defaults to None.
        start_flag_index (Optional[int], optional): Index where binary flags begin. 
            - Features before this index are scaled.
            - If None, all features are scaled. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
            - Scaled training dataset (`X_train_scaled`).
            - Scaled test dataset (`X_test_scaled`) or None if `test` is not provided.
            - Scaled validation dataset (`X_val_scaled`) or None if `val` is not provided.

    Example:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> scaler = MinMaxScaler()
        >>> X_train_scaled, X_test_scaled, X_val_scaled = scale_features(scaler, X_train, X_test, X_val, start_flag_index=5)
    """
    # Extract shape
    num_train, seq_len, num_feats = train.shape

    # Initialize placeholders
    X_test_scaled, X_val_scaled = None, None

    # Process TRAIN (fit & transform)
    if start_flag_index:
        X_train_features = train[:, :, :start_flag_index]  # Features to scale
        X_train_flags = train[:, :, start_flag_index:]  # Flags (unchanged)

        X_train_features_reshaped = X_train_features.reshape(-1, start_flag_index)
        X_train_features_scaled_2d = scaler.fit_transform(X_train_features_reshaped)
        X_train_features_scaled = X_train_features_scaled_2d.reshape(num_train, seq_len, start_flag_index)

        X_train_scaled = np.concatenate([X_train_features_scaled, X_train_flags], axis=2)
    else:
        X_train_reshaped = train.reshape(-1, num_feats)
        X_train_scaled_2d = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_2d.reshape(num_train, seq_len, num_feats)

    # Process test (transform only)
    if test is not None:
        num_test = test.shape[0]
        
        if start_flag_index:
            X_test_features = test[:, :, :start_flag_index]
            X_test_flags = test[:, :, start_flag_index:]

            X_test_features_reshaped = X_test_features.reshape(-1, start_flag_index)
            X_test_features_scaled_2d = scaler.transform(X_test_features_reshaped)
            X_test_features_scaled = X_test_features_scaled_2d.reshape(num_test, seq_len, start_flag_index)

            X_test_scaled = np.concatenate([X_test_features_scaled, X_test_flags], axis=2)
        else:
            X_test_reshaped = test.reshape(-1, num_feats)
            X_test_scaled_2d = scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled_2d.reshape(num_test, seq_len, num_feats)

    # Process validation (transform only)
    if val is not None:
        num_val = val.shape[0]

        if start_flag_index:
            X_val_features = val[:, :, :start_flag_index]
            X_val_flags = val[:, :, start_flag_index:]

            X_val_features_reshaped = X_val_features.reshape(-1, start_flag_index)
            X_val_features_scaled_2d = scaler.transform(X_val_features_reshaped)
            X_val_features_scaled = X_val_features_scaled_2d.reshape(num_val, seq_len, start_flag_index)

            X_val_scaled = np.concatenate([X_val_features_scaled, X_val_flags], axis=2)
        else:
            X_val_reshaped = val.reshape(-1, num_feats)
            X_val_scaled_2d = scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled_2d.reshape(num_val, seq_len, num_feats)

    return X_train_scaled, X_test_scaled, X_val_scaled

def scale_targets(
    scaler: TransformerMixin, 
    train: np.ndarray, 
    test: Optional[np.ndarray] = None, 
    val: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Scales target variables using a given scaler.

    This function scales target variables (`y`) for training, testing, and validation sets.
    The target values are flattened before scaling to ensure proper transformation.

    Args:
        scaler (TransformerMixin): A scikit-learn scaler instance (e.g., MinMaxScaler, StandardScaler).
        train (np.ndarray): Training target values with shape `(num_samples, horizon)`.
        test (Optional[np.ndarray], optional): Testing target values with the same shape as `train`. Defaults to None.
        val (Optional[np.ndarray], optional): Validation target values with the same shape as `train`. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
            - `y_train_scaled` (np.ndarray): Scaled training target values.
            - `y_test_scaled` (Optional[np.ndarray]): Scaled testing target values, or `None` if `test` is not provided.
            - `y_val_scaled` (Optional[np.ndarray]): Scaled validation target values, or `None` if `val` is not provided.

    Example:
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> scaler = MinMaxScaler()
        >>> y_train_scaled, y_test_scaled, y_val_scaled = scale_targets(scaler, y_train, y_test, y_val)
    """

    # Flatten target data for scaling
    y_train_flat = train.reshape(-1, 1)
    y_train_scaled_flat = scaler.fit_transform(y_train_flat)
    y_train_scaled = y_train_scaled_flat.reshape(train.shape)

    y_test_scaled, y_val_scaled = None, None

    if test is not None:
        y_test_flat = test.reshape(-1, 1)
        y_test_scaled_flat = scaler.transform(y_test_flat)
        y_test_scaled = y_test_scaled_flat.reshape(test.shape)

    if val is not None:
        y_val_flat = val.reshape(-1, 1)
        y_val_scaled_flat = scaler.transform(y_val_flat)
        y_val_scaled = y_val_scaled_flat.reshape(val.shape)

    return y_train_scaled, y_test_scaled, y_val_scaled

def undo_transformation(
    scaler: TransformerMixin, 
    y_scaled: np.ndarray
) -> np.ndarray:
    """Reverts the scaling and log transformation of target values.

    This function reverses the scaling transformation applied to `y` using `scaler.inverse_transform()`
    and then applies an exponential transformation to undo a previous log transformation.

    Args:
        scaler (TransformerMixin): A scikit-learn scaler instance used for the original scaling.
        y_scaled (np.ndarray): Scaled target values with shape `(num_samples, horizon)`.

    Returns:
        np.ndarray: The original (unscaled) target values, with log transformation reversed.

    Example:
        >>> y_original = undo_transformation(scaler, y_scaled)
    """

    # Undo scaling
    y_scaled_flat = y_scaled.reshape(-1, 1)
    y_flat = scaler.inverse_transform(y_scaled_flat)

    # Undo log transformation
    y = np.exp(y_flat)

    # Reshape to match original dimensions
    y = y.reshape(y_scaled.shape)
    
    return y
