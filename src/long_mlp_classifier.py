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
    """
    Preprocess the input data by handling missing values and converting all entries to numeric types.
    
    This function replaces missing values (denoted by '?') with NaN, coerces all data to numeric types,
    and fills NaN values with 0. The preprocessed data is returned as a NumPy array.

    Args:
        X (np.ndarray): The input data to be preprocessed.

    Returns:
        np.ndarray: The preprocessed data.
    """
    X_df: pd.DataFrame = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.fillna(0, inplace=True)
    return X_df.values

def load_and_preprocess_data() -> tuple:
    """
    Load and preprocess the longitudinal dataset.
    
    This function loads the dataset, splits it into training and test sets, preprocesses the features,
    and normalizes the data using Min-Max scaling. It also sets the global INPUT_SIZE variable and
    returns the preprocessed training and test sets, target values, and feature groups.

    Returns:
        tuple: A tuple containing the preprocessed training data, test data, training targets, test targets, and feature groups.
    """
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

class LongitudinalMLPModule(Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]]) -> None:
        """
        Initialize the Multi-Layer Perceptron (MLP) model designed for longitudinal data.
        
        This constructor initializes the model with hidden layers for each feature group and 
        sets the dropout rate for regularization. Each feature group is processed by a separate 
        hidden layer, and their outputs are concatenated before being passed to the final output layer.

        Args:
            hidden_size (int): The number of neurons in each hidden layer.
            output_size (int): The number of neurons in the output layer.
            dropout_rate (float): The dropout rate used for regularization.
            features_group (List[List[int]]): A list of feature groups, where each group contains indices of features.
        """
        super(LongitudinalMLPModule, self).__init__()
        self.features_group: List[List[int]] = features_group
        self.hidden_layers: ModuleList = ModuleList()
        self.dropout_rate: float = dropout_rate

        for group in features_group:
            group_size: int = len(group)
            self.hidden_layers.append(Sequential(
                Linear(group_size, hidden_size),
                Sigmoid()
            ))

        self.output_layer: Linear = Linear(hidden_size * len(features_group), output_size) 

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass through the MLP model.
        
        The input tensor is split into groups of features, which are then processed separately by
        corresponding hidden layers. The outputs of these layers are concatenated and passed through
        a dropout layer before being passed to the final output layer to generate the model's predictions.

        Args:
            X (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        group_outputs: List[Tensor] = []

        for idx, group in enumerate(self.features_group):
            group_x: Tensor = X[:, group]
            group_output: Tensor = self.hidden_layers[idx](group_x)
            group_outputs.append(group_output)

        concatenated_outputs: Tensor = cat(group_outputs, dim=1)
        concatenated_outputs = dropout(concatenated_outputs, p=self.dropout_rate, training=self.training)
        output: Tensor = self.output_layer(concatenated_outputs)

        return output

def create_search_space() -> dict:
    lr_range: List[float] = [0.3, 0.5, 0.6]
    max_epochs_range: List[int] = [200 ]
    hidden_size_range: List[int] = [32, 128]
    dropout_rate_range: List[float] = [0.3, 0.4]

    search_space: dict = {
        'lr': lr_range,
        'max_epochs': max_epochs_range,
        'module__hidden_size': hidden_size_range,
        'module__dropout_rate': dropout_rate_range,
    }

    return search_space

def train_and_evaluate_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, features_group: List[List[int]]) -> None:
    """
    Train the MLP model and evaluate its performance on the test data.
    
    This function initializes the model with the specified hyperparameters, trains it on the training
    data, and evaluates its performance on the test data by calculating various metrics such as 
    accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC score. The results are logged
    for further analysis.
    
    The hyperparameters are optimized using GridSearchCV, and the best model is selected based on the
    accuracy score.

    Args:
        X_train (np.ndarray): The preprocessed training data.
        X_test (np.ndarray): The preprocessed test data.
        y_train (np.ndarray): The training target values.
        y_test (np.ndarray): The test target values.
        features_group (List[List[int]]): A list of feature groups, where each group contains indices of features.
    """
    logger.info("Starting model training...")
    best_model: NeuralNetBinaryClassifier = NeuralNetBinaryClassifier(
        LongitudinalMLPModule(HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT_RATE, features_group),
        max_epochs=MAX_EPOCHS,
        lr=LR,
        iterator_train__shuffle=ITERATOR_TRAIN_SHUFFLE,
        train_split=TRAIN_SPLIT,
        verbose=VERBOSE,
        criterion=CRITERION
    )
    
    best_model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

    logger.info("Evaluating model...")
    y_predictions: np.ndarray  = best_model.predict(X_test.astype(np.float32))
    accuracy: float = np.mean(y_predictions == y_test)
    logger.info(f'Accuracy: {accuracy:.4f}')

    # Calculate additional metrics
    
    precision, recall, _ = precision_recall_curve(y_test, y_predictions)
    auprc = auc(recall, precision)
    
    precision: float = precision_score(y_test, y_predictions)
    recall: float = recall_score(y_test, y_predictions)
    f1: float = f1_score(y_test, y_predictions)
    conf_matrix: np.ndarray = confusion_matrix(y_test, y_predictions)
    roc_auc: float = roc_auc_score(y_test, y_predictions)
    report: classification_report = classification_report(y_test, y_predictions)
    

    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')
    logger.info(f'ROC-AUC Score: {roc_auc:.4f}')
    logger.info(f"Model target: {TARGET_WAVE}")
    #auprc = auprc_score(recall, precision)
    logger.info(f"AUPRC Score: {auprc}")
    logger.info(f"Classification Report: {report}")

def main() -> None:
    """
    The main function orchestrates the loading, preprocessing, training, and evaluation processes.
    
    This function serves as the entry point of the script, calling functions to load and preprocess
    the data, train the MLP model, and evaluate its performance.
    """
    X_train, X_test, y_train, y_test, features_group = load_and_preprocess_data()
    train_and_evaluate_model(X_train, X_test, y_train, y_test, features_group)

if __name__ == "__main__": 
    main()