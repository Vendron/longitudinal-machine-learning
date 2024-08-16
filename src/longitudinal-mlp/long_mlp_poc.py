from typing import List, Optional
from overrides import override
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from scikit_longitudinal.data_preparation import LongitudinalDataset
from functools import wraps
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
    Preprocesses the input data array by replacing '?' with NaN and converting to numeric.

    Args:
        X (np.ndarray): The input data array.

    Returns:
        np.ndarray: The preprocessed data as a NumPy array.
    """
    X_df: pd.DataFrame = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.fillna(0, inplace=True)
    
    return X_df.values

X_train: np.ndarray = preprocess_data(X_train)
X_test: np.ndarray = preprocess_data(X_test)

# Normalize data
scaler: MinMaxScaler = MinMaxScaler()
X_train: np.ndarray = scaler.fit_transform(X_train)
X_test: np.ndarray = scaler.transform(X_test)

# Ensure y_train and y_test are NumPy arrays and numeric
y_train = pd.to_numeric(y_train, errors='coerce').values
y_test = pd.to_numeric(y_test, errors='coerce').values

# This state validation is code from Scikit-Longitudinal package
def ensure_valid_state(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if method.__name__ in ["_predict", "_predict_proba"] and self._mlp is None:
            raise ValueError("Model has not been fitted yet. Call the fit method first before calling predict or predict_proba.")
        
        if hasattr(self, "features_group") and (self.features_group is None or len(self.features_group) <= 1):
            raise ValueError("Feature groups have not been set. Call the setup_features_group method first before calling fit.")
        
        return method(self, *args, **kwargs)
    return wrapper

class BaseMLP(BaseEstimator, ClassifierMixin):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLP":
        return self._fit(X, y)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLP":
        raise NotImplementedError("fit method not implemented")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict method not implemented")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_proba(X)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict_proba method not implemented")


# MLP implementation
class MLP(BaseMLP):
    """_summary_
    # Longitudinal Multi-layer Perceptron

    Here I want to test a simple MLP model on longitudinal data. 
    The model uses feedforward, with a single hidden layer, using a sigmoid activation function. The model is trained using backpropagation.
    - The number of neural units in the input layer is equal to the number of features in the dataset. 
    - The number of units in the hidden layer is a hyperparameter.
    - The number of units in the output layer is 1, as this is a binary classification problem. (Either patient is or is not diagnosed.)

    Returns:
        _type_: MLP
    """
    def __init__(self: object, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]]) -> None:
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size
        self.dropout_rate: float = dropout_rate
        self.features_group = features_group
        
        # Initialize weights for each feature group
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for group in features_group:
            group_size: int = len(group)
            self.weights.append(np.random.randn(group_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
        
        self.W2: np.ndarray = np.random.randn(hidden_size * len(features_group), output_size)
        self.b2: np.ndarray = np.zeros((1, output_size))
        self._mlp: Optional[BaseMLP] = None
        self.a1: Optional[np.ndarray] = None  # Store intermediate activations
        
    def sigmoid(self: object, z: float or np.ndarray) -> float or np.ndarray: # type: ignore
        """Sigmoid activation function

        Args:
            z (float or array-like): The input value(s) to the sigmoid function

        Returns:
            float or array-like: The output value(s) after applying the sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self: object, z: float or np.ndarray) -> float or np.ndarray: # type: ignore
        """Calculate derivative of sigmoid function

        Args:
            self (object): The object instance.
            z (float or np.ndarray): The input value(s) to the sigmoid function

        Returns:
            float or np.ndarray: Derivative of the sigmoid function
        """
        return z * (1 - z)
    
    def dropout(self: object, layer_output: np.ndarray) -> np.ndarray:
        """Apply dropout regularization to the layer output.

        Args:
            self (object): The instance of the class.
            layer_output (np.ndarray): The output of the layer.

        Returns:
            np.ndarray: The layer output after applying dropout regularization.
        """
        mask: np.ndarray = (np.random.rand(*layer_output.shape) < self.dropout_rate) / self.dropout_rate 
        return layer_output * mask
    
    def forward(self: object, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Calculates the forward pass of the neural network.

        Args:
            self (object): The neural network object.
            X (np.ndarray): The input data.
            training (bool): Whether the forward pass is performed during training or not.

        Returns:
            np.ndarray: The output of the neural network.
        """
        group_outputs: List[np.ndarray] = []
        self.a1: List[np.ndarray] = []
        for idx, group in enumerate(self.features_group):
            group_x: np.ndarray = X[:, group]
            z1: np.ndarray = np.dot(group_x, self.weights[idx]) + self.biases[idx]
            a1: np.ndarray = self.sigmoid(z1)
            if training:
                a1 = self.dropout(a1)
            group_outputs.append(a1)
            self.a1.append(a1)
        
        concatenated_outputs: np.ndarray = np.concatenate(group_outputs, axis=1)
        
        # Final output layer
        z2: np.ndarray = np.dot(concatenated_outputs, self.W2) + self.b2
        a2: np.ndarray = self.sigmoid(z2)
        
        return a2
    
    def compute_loss(self: object, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the loss between predicted and true values.

        Args:
            self (object): The object instance.
            y_pred (np.ndarray): The predicted values.
            y_true (np.ndarray): The true values.

        Returns:
            float: The computed loss.
        """
        m: int = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10) # Avoid division by zero and log(0)
        loss: float = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self: object, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, learning_rate: float) -> None:
        """
        Update the weights and biases of the neural network using backpropagation.

        Args:
            self (object): The neural network object.
            X (np.ndarray): The input data.
            y (np.ndarray): The target labels.
            y_pred (np.ndarray): The predicted labels.
            learning_rate (float): The learning rate for updating the weights and biases.
        """
        m: int = y.shape[0]
        
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10) # Avoid division by zero
        
        # Compute gradients
        d_loss_a2: np.ndarray = -(y / y_pred) + ((1 - y) / (1 - y_pred))
        d_loss_z2: np.ndarray = d_loss_a2 * self.sigmoid_derivative(y_pred)
        
        d_loss_W2: np.ndarray = np.dot(np.concatenate(self.a1, axis=1).T, d_loss_z2) / m
        d_loss_b2: np.ndarray = np.sum(d_loss_z2, axis=0, keepdims=True) / m 
        
        d_loss_a1: List[np.ndarray] = np.dot(d_loss_z2, self.W2.T)
        d_loss_z1: List[np.ndarray] = [d_loss_a1[:, i * self.hidden_size:(i + 1) * self.hidden_size] * self.sigmoid_derivative(self.a1[i])
                     for i in range(len(self.features_group))]
        
        d_loss_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        d_loss_biases: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        
        for idx, group in enumerate(self.features_group):
            group_x: np.ndarray = X[:, group]
            d_loss_weights[idx]: np.ndarray = np.dot(group_x.T, d_loss_z1[idx]) / m
            d_loss_biases[idx]: np.ndarray = np.sum(d_loss_z1[idx], axis=0, keepdims=True) / m
        
        # Update the weights and biases
        for idx in range(len(self.weights)):
            self.weights[idx] -= learning_rate * d_loss_weights[idx]
            self.biases[idx] -= learning_rate * d_loss_biases[idx]
        
        self.W2 -= learning_rate * d_loss_W2
        self.b2 -= learning_rate * d_loss_b2
        
    def calc_loss(self: object, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate the loss between predicted and true values.

        Args:
            self (object): The object instance.
            y_pred (np.ndarray): The predicted values.
            y_true (np.ndarray): The true values.

        Returns:
            float: The calculated loss.
        """
        m: int = y_true.shape[0]
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10) # Avoid division by zero
        loss: float = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
        
    @ensure_valid_state
    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """Fits the MLP model to the given training data.

        Args:
            X (np.ndarray): The input features of shape (n_samples, n_features).
            y (np.ndarray): The target values of shape (n_samples,).

        Returns:
            MLP: The fitted MLP model.
        """
        self._mlp = self
        for epoch in range(epochs):
            y_pred = self.forward(X, training=True)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y, y_pred, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        return self

    @ensure_valid_state
    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the output for the given input.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        y_pred = self.forward(X, training=False)
        return np.round(y_pred)
    
    @ensure_valid_state
    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the probabilities of the target classes for the input data.

        Args:
            X (np.ndarray): The input data to be used for prediction.

        Returns:
            np.ndarray: The predicted probabilities of the target classes.
        """
        return self.forward(X, training=False)
    
# Model Parameters
hidden_size: int = 64
output_size: int = 1
epochs: int = 1000
learning_rate: float = 0.01
dropout_rate: float = 0.5
features_group: List[List[int]] = dataset.feature_groups()

mlp: object = MLP(hidden_size, output_size, dropout_rate, features_group)
mlp.fit(X_train, y_train.reshape(-1, 1))

# Predict and evaluate
y_pred: np.ndarray = mlp.predict(X_test)
accuracy: float = np.mean(y_pred == y_test.reshape(-1, 1))
print(f'Accuracy: {accuracy}')

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