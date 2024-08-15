from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from scikit_longitudinal.data_preparation import LongitudinalDataset
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, average_precision_score, roc_curve, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH: str = os.getenv("DATASET_PATH")
TARGET_WAVE: str = os.getenv("TARGET_WAVE")
# Hyperparameters
INPUT_SIZE: int = None  # Will be set after loading data 
HIDDEN_SIZE: int = 128  # Number of neurons in each hidden layer
MAX_EPOCHS: int = 300  # Increased maximum number of epochs for training
LR: float = 0.001  # Adjusted learning rate for optimization
DROPOUT_RATE: float = 0.4  # Dropout rate for regularization
KFOLDS: int = 10  # Number of folds for cross-validation

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def preprocess_data(X: np.ndarray) -> np.ndarray:
    X_df: pd.DataFrame = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.fillna(0, inplace=True)
    return X_df.values

def load_and_preprocess_data() -> tuple:
    dataset: LongitudinalDataset = LongitudinalDataset(DATASET_PATH)
    dataset.load_data_target_train_test_split(target_column=TARGET_WAVE, random_state=42, test_size=0.1, remove_target_waves=True)
    dataset.setup_features_group("elsa")

    X_train: np.ndarray = preprocess_data(dataset.X_train)
    X_test: np.ndarray = preprocess_data(dataset.X_test)
    y_train: np.ndarray = pd.to_numeric(dataset.y_train, errors='coerce').values
    y_test: np.ndarray = pd.to_numeric(dataset.y_test, errors='coerce').values

    # Normalize data
    scaler: MinMaxScaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    global INPUT_SIZE
    INPUT_SIZE = X_train.shape[1]

    return X_train, X_test, y_train, y_test, dataset.feature_groups()

def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    logger.info("Starting model training...")

    # Define the MLP Classifier with modifications
    mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        activation='relu',
        solver='adam',
        max_iter=MAX_EPOCHS,
        learning_rate_init=LR,
        alpha=DROPOUT_RATE,
        random_state=42,
        early_stopping=True,  # Enable early stopping
        n_iter_no_change=10   # Number of epochs with no improvement to wait before stopping
    )

    # Perform Grid Search with KFold Cross-Validation
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    param_grid = {
        'hidden_layer_sizes': [(64,), (HIDDEN_SIZE,)],
        'learning_rate_init': [0.01, LR],
        'alpha': [0.001, DROPOUT_RATE]
    }

    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='f1', cv=kf, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_mlp = grid_search.best_estimator_
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # Evaluate the model
    y_predictions = best_mlp.predict(X_test)
    accuracy = np.mean(y_predictions == y_test)
    logger.info(f'Accuracy: {accuracy:.4f}')

    # Calculate additional metrics
    precision = precision_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions)
    f1 = f1_score(y_test, y_predictions)
    conf_matrix = confusion_matrix(y_test, y_predictions)
    roc_auc = roc_auc_score(y_test, y_predictions)

    # Calculate precision-recall curve and AUPRC
    y_prob = best_mlp.predict_proba(X_test)[:, 1]
    precision_recall = precision_recall_curve(y_test, y_prob)
    auprc = auc(precision_recall[1], precision_recall[0])
    report = classification_report(y_test, y_predictions)

    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')
    logger.info(f'ROC-AUC Score: {roc_auc:.4f}')
    logger.info(f"AUPRC Score: {auprc:.4f}")
    logger.info(f"Classification Report: {report}")

def main() -> None:
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__": 
    main()
