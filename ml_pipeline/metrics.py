"""
metrics.py

This script provides functions to evaluate and visualize the performance of a forecasting model 
using various error metrics, including Symmetric Mean Absolute Percentage Error (sMAPE), 
Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). It also generates plots to 
analyze the distribution of errors over multiple predictions.

Functions:
    - smape(true_values, pred_values): Computes the sMAPE for a given set of true and predicted values.
    - calculate_metrics(model, true_values, pred_values, save=False): 
        Calculates RMSE, MAE, and sMAPE for a given model's predictions, prints summary statistics, 
        and saves results if required.
    - plot_density_comparison(true_values, pred_values, plot_name, save): 
        Generates and optionally saves a density comparison plot between true and predicted values.
    - plot_rmse_over_predictions(per_pred_rmse_values, plot_name, save): 
        Plots RMSE over a series of predictions.
    - plot_mae_over_predictions(per_pred_mae_values, plot_name, save): 
        Plots MAE over a series of predictions.
    - plot_smape_over_predictions(per_pred_smape_values, plot_name, save): 
        Plots sMAPE over a series of predictions.

Usage:
    This script can be used to evaluate the accuracy of a predictive model's outputs.
    Call `calculate_metrics(model, true_values, pred_values, save=True/False)` 
    to compute error metrics and generate plots.

"""

# Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from typing import Any

def smape(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """
    Computes the Symmetric Mean Absolute Percentage Error (sMAPE) between true and predicted values.

    sMAPE is a percentage-based error metric that measures the relative difference between 
    actual and predicted values while ensuring symmetry between over- and under-predictions.

    Args:
        true_values (np.ndarray): A NumPy array of actual values.
        pred_values (np.ndarray): A NumPy array of predicted values.

    Returns:
        float: The sMAPE value, expressed as a percentage.

    Example:
        >>> import numpy as np
        >>> true = np.array([100, 200, 300])
        >>> pred = np.array([110, 190, 290])
        >>> smape(true, pred)
        3.1746031746031744
    """
    # Ensure input values are NumPy arrays of type float64
    true = np.array(true_values, dtype=np.float64)
    pred = np.array(pred_values, dtype=np.float64)
    
    # Calculate the absolute difference (numerator)
    numerator = np.abs(true - pred)
    
    # Compute the denominator as the mean absolute values of true and predicted values
    denominator = (np.abs(true) + np.abs(pred)) / 2
    
    # Handle cases where both true and predicted values are zero to avoid division errors
    mask = (true == 0) & (pred == 0)
    
    # Compute sMAPE values, ensuring zero error when both true and predicted values are zero
    smape_values = np.where(mask, 0.0, numerator / denominator)
    
    # Return the mean sMAPE value as a percentage
    return np.mean(smape_values) * 100

def calculate_metrics(model: Any, true_values: list[np.ndarray], pred_values: list[np.ndarray], save: bool = False) -> None:
    """
    Calculates and visualizes key error metrics for model predictions, including RMSE, MAE, and sMAPE.

    This function computes:
    - Per-prediction and average Root Mean Squared Error (RMSE).
    - Per-prediction and average Mean Absolute Error (MAE).
    - Per-prediction and average Symmetric Mean Absolute Percentage Error (sMAPE).
    - Minimum, maximum, and mean values for both true and predicted data.
    
    It also generates plots to analyze errors over predictions and can save the results and visualizations.

    Args:
        model (Any): The predictive model being evaluated. If the model has a `summary()` method, it will be written to the results file.
        true_values (list[np.ndarray]): A list of NumPy arrays representing the actual values for multiple predictions.
        pred_values (list[np.ndarray]): A list of NumPy arrays representing the predicted values for multiple predictions.
        save (bool, optional): Whether to save the results and plots to files. Defaults to False.

    Returns:
        None
    """
    # Calculate per-prediction RMSE
    per_pred_rmse_values = [
        root_mean_squared_error(np.array(actual), np.array(predicted)) 
        for actual, predicted in zip(true_values, pred_values)
    ]
    average_rmse = np.mean(per_pred_rmse_values)
    plot_rmse_over_predictions(np.array(per_pred_rmse_values), f"{average_rmse}.png", save)

    # Calculate per-prediction MAE
    per_pred_mae_values = [
        np.mean(np.abs(np.array(actual) - np.array(predicted))) 
        for actual, predicted in zip(true_values, pred_values)
    ]
    average_mae = np.mean(per_pred_mae_values)
    plot_mae_over_predictions(np.array(per_pred_mae_values), f"{average_mae}.png", save)

    # Calculate per-prediction SMAPE
    per_pred_smape_values = [
        smape(np.array(actual), np.array(predicted)) 
        for actual, predicted in zip(true_values, pred_values)
    ]
    average_smape = np.mean(per_pred_smape_values)
    plot_smape_over_predictions(np.array(per_pred_smape_values), f"{average_smape}.png", save)

    # Convert lists to NumPy arrays to avoid flatten errors
    all_true_values = np.concatenate([np.array(arr) for arr in true_values])
    all_pred_values = np.concatenate([np.array(arr) for arr in pred_values])

    # Calculate overall min, max, and mean values
    min_true, min_pred = np.min(all_true_values), np.min(all_pred_values)
    max_true, max_pred = np.max(all_true_values), np.max(all_pred_values)
    mean_true, mean_pred = np.mean(all_true_values), np.mean(all_pred_values)

    # Print metrics summary
    print(f"Average Symmetric Mean Absolute Percentage Error (SMAPE): {average_smape}%")
    print(f"Average Root Mean Squared Error (RMSE): {average_rmse}")
    print(f"Average Mean Absolute Error (MAE): {average_mae}")
    print(f"Min predicted: {min_pred}, Min actual: {min_true}")
    print(f"Max predicted: {max_pred}, Max actual: {max_true}")
    print(f"Mean predicted: {mean_pred}, Mean actual: {mean_true}\033[0m")

    # Generate valid file path
    file_path = f"metrics/{average_smape}%_{average_rmse}_{average_mae}.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists

    # Extract filename correctly for density comparison plot
    plot_name = ''.join(os.path.basename(file_path).split('.')[:-1])
    plot_density_comparison(all_true_values, all_pred_values, plot_name=plot_name, save=save)

    # Save metrics to a file if requested
    if save:
        with open(file_path, "w") as file:
            file.write(f"Average SMAPE: {average_smape}%\n")
            file.write(f"Average RMSE: {average_rmse}\n")
            file.write(f"Average MAE: {average_mae}\n")
            file.write(f"Min predicted: {min_pred}, Min actual: {min_true}\n")
            file.write(f"Max predicted: {max_pred}, Max actual: {max_true}\n")
            file.write(f"Mean predicted: {mean_pred}, Mean actual: {mean_true}\n\n")

            # Handle model summary safely
            if hasattr(model, "summary"):
                model.summary(print_fn=lambda x: file.write(x + "\n"))
            else:
                file.write("Model summary not available.\n")

def plot_density_comparison(true_values: np.ndarray, pred_values: np.ndarray, plot_name: str, save: bool) -> None:
    """
    Generates and optionally saves a density comparison plot between true and predicted values.

    This function visualizes how well the predicted values align with the actual values 
    by plotting them against their respective indices. If `save` is enabled, the plot 
    is saved to the 'plots/' directory with the given `plot_name`.

    Args:
        true_values (np.ndarray): A NumPy array of actual values.
        pred_values (np.ndarray): A NumPy array of predicted values.
        plot_name (str): The filename for saving the plot (without extension).
        save (bool): Whether to save the plot as an image file.

    Returns:
        None
    """
    # Define the file path for saving the plot
    file_path = f'plots/{plot_name}'

    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Plot actual and predicted values
    plt.plot(range(len(true_values)), true_values, label='True Values', color='blue')
    plt.plot(range(len(pred_values)), pred_values, label='Predicted Values', color='orange')
    
    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Orbit Mean Density (kg/m³)')
    plt.title('Predicted vs Actual Values')
    plt.legend()

    # Save the plot if required
    if save:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_rmse_over_predictions(per_pred_rmse_values: np.ndarray, plot_name: str, save: bool) -> None:
    """
    Plots Root Mean Squared Error (RMSE) over a series of predictions.

    This function visualizes RMSE values across multiple predictions, helping to 
    assess how the error evolves over time. If `save` is enabled, the plot is saved 
    to the 'plots/rmse/' directory with the specified `plot_name`.

    Args:
        per_pred_rmse_values (np.ndarray): A NumPy array containing RMSE values for each prediction.
        plot_name (str): The filename for saving the plot (without extension).
        save (bool): Whether to save the plot as an image file.

    Returns:
        None
    """
    # Define the file path for saving the plot
    file_path = f"plots/rmse/{plot_name}"

    # Set figure size
    plt.figure(figsize=(12, 6))

    # Plot RMSE values over the prediction period
    plt.plot(range(len(per_pred_rmse_values)), per_pred_rmse_values, label='RMSE Values', color='blue')

    # Add labels and title
    plt.xlabel('Prediction Number')
    plt.ylabel('RMSE For Prediction')
    plt.title('RMSE over Series of Predictions')
    plt.legend()

    # Save the plot if required
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_mae_over_predictions(per_pred_mae_values: np.ndarray, plot_name: str, save: bool) -> None:
    """
    Plots Mean Absolute Error (MAE) over a series of predictions.

    This function visualizes MAE values across multiple predictions to evaluate the consistency 
    of prediction errors. If `save` is enabled, the plot is saved to the 'plots/mae/' directory 
    with the specified `plot_name`.

    Args:
        per_pred_mae_values (np.ndarray): A NumPy array containing MAE values for each prediction.
        plot_name (str): The filename for saving the plot (without extension).
        save (bool): Whether to save the plot as an image file.

    Returns:
        None
    """
    # Define the file path for saving the plot
    file_path = f"plots/mae/{plot_name}"

    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Plot MAE values over the prediction period
    plt.plot(range(len(per_pred_mae_values)), per_pred_mae_values, label='MAE Values', color='blue')
    
    # Add labels and title
    plt.xlabel('Prediction Number')
    plt.ylabel('MAE For Prediction')
    plt.title('MAE over Series of Predictions')
    plt.legend()

    # Save the plot if required
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_smape_over_predictions(per_pred_smape_values: np.ndarray, plot_name: str, save: bool) -> None:
    """
    Plots Symmetric Mean Absolute Percentage Error (sMAPE) over a series of predictions.

    This function visualizes sMAPE values across multiple predictions to assess the error trends 
    in a model’s predictions. If `save` is enabled, the plot is saved to the 'plots/smape/' directory 
    with the specified `plot_name`.

    Args:
        per_pred_smape_values (np.ndarray): A NumPy array containing sMAPE values for each prediction.
        plot_name (str): The filename for saving the plot (without extension).
        save (bool): Whether to save the plot as an image file.

    Returns:
        None
    """
    # Define the file path for saving the plot
    file_path = f"plots/smape/{plot_name}"

    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Plot sMAPE values over the prediction period
    plt.plot(range(len(per_pred_smape_values)), per_pred_smape_values, label='sMAPE Values', color='blue')
    
    # Add labels and title
    plt.xlabel('Prediction Number')
    plt.ylabel('sMAPE For Prediction')
    plt.title('sMAPE over Series of Predictions')
    plt.legend()

    # Save the plot if required
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
