from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from data_preparation.preprocessing import LongitudinalDataset
from models.mlp import MLP

# Clear any cached environment variables
os.environ.pop("DATASET_PATH", None)
os.environ.pop("TARGET_WAVE", None)

# Load environment variables
env_path = find_dotenv()
print(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Get variables from environment
DATASET_PATH = os.getenv("DATASET_PATH")
TARGET_WAVE = os.getenv("TARGET_WAVE")

# Debug print statements to verify the environment variables
print(f"DATASET_PATH: '{DATASET_PATH}'")
print(f"TARGET_WAVE: '{TARGET_WAVE}'")

# Validate dataset path
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# Load and prepare the dataset
dataset = LongitudinalDataset(DATASET_PATH)
dataset.load_data_target_train_test_split(target_column=TARGET_WAVE, random_state=42)
dataset.setup_features_group("elsa")

X_train, X_test, y_train, y_test = dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test

# Preprocess the data
def preprocess_data(X: np.ndarray) -> np.ndarray:
    """
    Preprocesses the input data array by replacing '?' with NaN and converting to numeric.

    Args:
        X (np.ndarray): The input data array.

    Returns:
        np.ndarray: The preprocessed data as a NumPy array.
    """
    X_df = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.fillna(0, inplace=True)
    return X_df.values

X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert y to numeric values
y_train = pd.to_numeric(y_train, errors='coerce').values
y_test = pd.to_numeric(y_test, errors='coerce').values

# Model parameters
hidden_size = 64
output_size = 1
epochs = 1000
learning_rate = 0.01
dropout_rate = 0.5
features_group = dataset.get_feature_groups()

# Initialize and train the model
mlp = MLP(hidden_size, output_size, dropout_rate, features_group, epochs, learning_rate)
mlp.fit(X_train, y_train.reshape(-1, 1))

# Predict and evaluate
y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test.reshape(-1, 1))
print(f'Accuracy: {accuracy}')

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'ROC-AUC Score: {roc_auc}')