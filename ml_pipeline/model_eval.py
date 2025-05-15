"""
model_eval.py

This script evaluates the accuracy of a density prediction model against a ground truth dataset.
The evaluation is based on multiple error metrics and a computed **Propagation Score (PS)**, 
which compares the current model's predictions to a benchmark "Best" model.

The script provides:
- Data validation and alignment between ground truth and participant predictions.
- Calculation of RMSE, SMAPE, and MAE metrics.
- Computation of the **Propagation Score (PS)** to assess predictive performance.

Dependencies:
    - numpy
    - pandas
    - json
    - os
    - warnings
    - metrics (custom module)

Classes:
    DensityModelEvaluator: Evaluates and scores a a model based on prediction accuracy.

Functions:
    run_evaluator(ground_truth_path, participant_path): Runs evaluation and computes the Propagation Score (PS).
    generate_score(path): Generates the Propagation Score (PS) for a given submission.

Example Usage:
    >>> ps = generate_score("submissions/submission.json")
"""

# Imports
import numpy as np
import pandas as pd
import json
import os
import warnings
from metrics import calculate_metrics

# Suppress warnings
warnings.filterwarnings("ignore")

class DensityModelEvaluator:
    """
    Evaluates the accuracy of a density prediction model using various error metrics and the Propagation Score (PS).

    Attributes:
        ground_truth (dict): Dictionary containing the ground truth dataset.
        participant (dict): Dictionary containing the participant's predictions.

    Methods:
        calculate_metrics(): Computes RMSE, SMAPE, and MAE for the participant's predictions.
        score(epsilon=1e-5): Computes the Propagation Score (PS).
    """

    def __init__(self, ground_truth_file: str, participant_file: str) -> None:
        """
        Initializes the evaluator with ground truth and participant data.

        Args:
            ground_truth_file (str): Path to the ground truth JSON file.
            participant_file (str): Path to the participant JSON file.
        
        Raises:
            ValueError: If the structure of the ground truth and participant data does not match.
        """
        self.ground_truth = self._load_json(ground_truth_file)
        self.participant = self._load_json(participant_file)

        # Validate data structure
        if not self._validate_data():
            raise ValueError("Mismatch between ground truth and participant data structure.")

    def _load_json(self, file_path: str) -> dict:
        """
        Loads and parses a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Parsed JSON data.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def _validate_data(self) -> bool:
        """
        Validates that the ground truth and participant data contain the same keys.

        Returns:
            bool: True if the structures match, False otherwise.
        """
        return set(self.ground_truth.keys()) == set(self.participant.keys())

    def _pad_or_truncate(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """
        Adjusts the length of an array by padding with NaNs or truncating.

        Args:
            array (np.ndarray): Input array.
            target_length (int): Desired length of the array.

        Returns:
            np.ndarray: Adjusted array with length equal to target_length.
        """
        current_length = len(array)
        if current_length == target_length:
            return array
        elif current_length < target_length:
            padding = np.full(target_length - current_length, np.nan)
            return np.concatenate([array, padding])
        else:
            return array[:target_length]

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Combines ground truth, Best model, and participant predictions into a single DataFrame.

        Returns:
            pd.DataFrame: Combined DataFrame containing all relevant data.
        """
        combined_data = []

        for file_id in self.ground_truth.keys():
            timestamps = pd.to_datetime(self.ground_truth[file_id]['Timestamp'], errors='coerce')
            num_timestamps = len(timestamps)

            truth_density = np.array(self.ground_truth[file_id]['Orbit Mean Density (kg/m^3)'])
            best_density = np.array(self.ground_truth[file_id]['Pred Orbit Mean Density (kg/m^3)'])
            pred_density = np.array(self.participant[file_id]['Pred Orbit Mean Density (kg/m^3)'])

            # Align array lengths
            truth_density = self._pad_or_truncate(truth_density, num_timestamps)
            best_density = self._pad_or_truncate(best_density, num_timestamps)
            pred_density = self._pad_or_truncate(pred_density, num_timestamps)

            # Create DataFrame
            file_df = pd.DataFrame({
                'FileID': [file_id] * num_timestamps,
                'Timestamp': timestamps,
                'TruthDensity': truth_density,
                'BestDensity': best_density,
                'PredictDensity': pred_density
            })

            file_df['DeltaTime'] = (file_df['Timestamp'] - file_df['Timestamp'].iloc[0]).dt.total_seconds()
            combined_data.append(file_df)

        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.replace(9.99e32, np.nan, inplace=True)
        combined_df.dropna(subset=['TruthDensity', 'BestDensity', 'PredictDensity'], inplace=True)

        return combined_df

    def calculate_metrics(self) -> None:
        """
        Computes and prints RMSE, SMAPE, and MAE for both the Best and Participant models.
        """
        true_predictions = [self.ground_truth[file_id]['Orbit Mean Density (kg/m^3)'] for file_id in self.ground_truth]
        best_predictions = [self.ground_truth[file_id]['Pred Orbit Mean Density (kg/m^3)'] for file_id in self.ground_truth]
        new_predictions = [self.participant[file_id]['Pred Orbit Mean Density (kg/m^3)'] for file_id in self.participant]

        true_predictions = np.array(true_predictions)
        best_predictions = np.array(best_predictions)
        new_predictions = np.array(new_predictions)

        print("\n\n\033[32mBest Submission Metrics: ")
        calculate_metrics(None, true_predictions, best_predictions, save=False)

        print("\n\n\033[32mNew Submission Metrics: ")
        calculate_metrics(None, true_predictions, new_predictions, save=True)

    def score(self, epsilon: float = 1e-5) -> float:
        """
        Computes the Propagation Score (PS) based on the provided data.

        Args:
            epsilon (float, optional): Minimum weight value at the end of the forecast period. Defaults to 1e-5.

        Returns:
            float: Calculated Propagation Score (PS), or np.nan if no valid data is available.
        """
        self.calculate_metrics()
        combined_df = self._prepare_dataframe()

        if combined_df.empty:
            return np.nan

        combined_df['Best_ErrorSq'] = (combined_df['BestDensity'] - combined_df['TruthDensity']) ** 2
        combined_df['Pred_ErrorSq'] = (combined_df['PredictDensity'] - combined_df['TruthDensity']) ** 2

        mse_grouped = combined_df.groupby('DeltaTime')[['Best_ErrorSq', 'Pred_ErrorSq']].mean()
        rmse_df = mse_grouped.apply(np.sqrt)
        rmse_df.columns = ['Best_RMSE', 'Pred_RMSE']

        delta_times = rmse_df.index.values
        total_duration = max(delta_times[-1] - delta_times[0], 1e-12)
        decay_rate = -np.log(epsilon) / total_duration
        weights = np.exp(-decay_rate * (delta_times - delta_times[0]))

        rmse_df['Best_RMSE'].replace(0, 1e-32, inplace=True)
        rmse_df['Skill'] = 1 - (rmse_df['Pred_RMSE'] / rmse_df['Best_RMSE'])

        return np.average(rmse_df['Skill'].values, weights=weights)

def run_evaluator(ground_truth_path: str, participant_path: str) -> float:
    """Runs the evaluation and computes the Propagation Score (PS)."""
    evaluator = DensityModelEvaluator(ground_truth_path, participant_path)
    ps = evaluator.score()
    print(f'\n\nPropagation Score (PS): {ps:.6f}')
    return ps

def generate_score(path: str = "", ground_truth_path = "submissions/best_submission.json") -> float:
    """Generates the Propagation Score (PS) for a given submission."""
    return run_evaluator(ground_truth_path, path)
