from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from scikit_longitudinal.data_preparation import LongitudinalDataset
from models.mlp.mlp import TemporalMLP
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Get variables from environment
DATASET_PATH: str = os.getenv("DATASET_PATH")
TARGET_WAVE: str = os.getenv("TARGET_WAVE")

dataset: LongitudinalDataset = LongitudinalDataset(DATASET_PATH)
dataset.load_data_target_train_test_split(target_column=TARGET_WAVE, random_state=42)
dataset.setup_features_group("elsa")

X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test

def preprocess_data(X: np.ndarray) -> np.ndarray:
    """
    Preprocesses the input data array by replacing '?' with NaN.

    Args:
        X (np.ndarray): The input data array.

    Returns:
        np.ndarray: The preprocessed data as a NumPy array.
    """
    X_df: pd.DataFrame = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df.fillna(0, inplace=True)
    return X_df.values

X_train: np.ndarray = preprocess_data(X_train)
X_test: np.ndarray = preprocess_data(X_test)

# Normalize data
scaler: MinMaxScaler = MinMaxScaler()
X_train: np.ndarray = scaler.fit_transform(X_train).astype(np.float32)
X_test: np.ndarray = scaler.transform(X_test).astype(np.float32)

# Convert y to numeric values (float32) because skorch requires float32
y_train: np.ndarray = pd.to_numeric(y_train, errors='coerce').values.astype(np.float32)
y_test: np.ndarray = pd.to_numeric(y_test, errors='coerce').values.astype(np.float32)

# Model parameters
input_size: int = X_train.shape[1]
hidden_sizes: List[int] = [64, 32]  # List of hidden layer sizes
output_size: int = 1
epochs: int = 250
learning_rate: float = 0.01
dropout_rate: float = 0.5
features_group: List[List[int]] = dataset.feature_groups()

# Initialize and train the model
mlp: TemporalMLP = TemporalMLP(input_size, hidden_sizes, output_size, dropout_rate, epochs, learning_rate, features_group)

# Reshape y_train to 2D array
y_train: np.ndarray = y_train.reshape(-1, 1)

kf: KFold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results: Dict[str, List[float]] = cross_validate(mlp, X_train, y_train, cv=kf, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

print(f'CV Results: {cv_results}')
print(f'Accuracy: {np.mean(cv_results["test_accuracy"])}')

mlp.fit(X_train, y_train)

# Predict and evaluate
y_pred: np.ndarray = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test.reshape(-1, 1))

# Calculate additional metrics
precision: float = precision_score(y_test, y_pred)
recall: float = recall_score(y_test, y_pred)
f1: float = f1_score(y_test, y_pred)
conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)
roc_auc: float = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'ROC-AUC Score: {roc_auc}')