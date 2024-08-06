from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from scikit_longitudinal.data_preparation import LongitudinalDataset
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Linear, Sigmoid, Sequential, BCEWithLogitsLoss, ModuleList
from torch import Tensor, cat
from torch.nn.functional import dropout
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, average_precision_score, roc_curve, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import logging
from functools import partial
from scikit_longitudinal.metrics import auprc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Constants
DATASET_PATH: str = os.getenv("DATASET_PATH")
TARGET_WAVE: str = os.getenv("TARGET_WAVE")
# Hyperparameters
INPUT_SIZE: int = None  # Will be set after loading data 
HIDDEN_SIZE: int = 128 # Number of neurons in each hidden layer
OUTPUT_SIZE: int = 1 # Number of neurons in the output layer
DROPOUT_RATE: float = 0.4 # Dropout rate for regularization (Dropout layer is applied after hidden layers to prevent overfitting by randomly setting a fraction of input units to 0)
# Specific hyperparameters for the MLP model
MAX_EPOCHS: int = 200 # Maximum number of epochs for training
LR: float = 0.3 # Learning rate for optimization (Adam optimizer is used by default, however other optimizers can be used by specifying the optimizer parameter in the NeuralNetBinaryClassifier constructor)
ITERATOR_TRAIN_SHUFFLE: bool = True # Shuffle the training data before each epoch
TRAIN_SPLIT: None = None # Use the entire training data for training
VERBOSE: int = 1 # Verbosity level for logging during training, where 0 = silent, 1 = progress bar, 2 = one line per epoch and 3 = one line per batch.
CRITERION: Module = BCEWithLogitsLoss() # Binary Cross-Entropy loss function for binary classification tasks. This function combines a Sigmoid layer and the BCE loss in one single class.
KFOLDS: int = 10 # Number of folds for cross-validation

# Set seeds for reproducibility - Note later: Issue with re-recurring confusion matrix values.  
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
    dataset.load_data_target_train_test_split(target_column=TARGET_WAVE, random_state=42, test_size=0.1)
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

    # Define the MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        activation='relu',
        solver='adam',
        max_iter=MAX_EPOCHS,
        learning_rate_init=LR,
        alpha=DROPOUT_RATE,
        random_state=42
    )

    # Evaluate the model
    best_mlp = mlp.fit(X_train, y_train)
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
    report: classification_report = classification_report(y_test, y_predictions)

    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')
    logger.info(f'ROC-AUC Score: {roc_auc:.4f}')
    logger.info(f"AUPRC Score: {auprc:.4f}")
    logger.info(f"Model target: {TARGET_WAVE}")
    logger.info(f"Classification Report: {report}")


def main() -> None:
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__": 
    main()