from typing import List, Optional
from overrides import override
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import wraps
from utils.logger import TrainingLogger

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
    """Base class for MLP models.

    This class provides a base implementation for MLP (Multi-Layer Perceptron) models.
    It inherits from `BaseEstimator` and `ClassifierMixin` classes.

    Args:
        BaseEstimator (type): Base class for all estimators in scikit-learn.
        ClassifierMixin (type): Mixin class for all classifiers in scikit-learn.
    """
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLP":
        """Fit the model to the training data.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target labels.

        Returns:
            BaseMLP: The fitted model instance.
        """
        return self._fit(X, y)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> "BaseMLP":
        """Internal method to fit the model to the training data.

        This method should be implemented by the derived classes.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target labels.

        Returns:
            BaseMLP: The fitted model instance.
        """
        raise NotImplementedError("fit method not implemented")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the input data.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self._predict(X)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal method to predict the class labels for the input data.

        This method should be implemented by the derived classes.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted class labels.
        """
        raise NotImplementedError("predict method not implemented")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the class probabilities for the input data.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        return self._predict_proba(X)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Internal method to predict the class probabilities for the input data.

        This method should be implemented by the derived classes.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        raise NotImplementedError("predict_proba method not implemented")

class MLP(BaseMLP):
    """
    Longitudinal Multi-layer Perceptron (MLP) model.

    This MLP model is designed for longitudinal data, using feedforward architecture with a single hidden layer
    and a sigmoid activation function. The model is trained using backpropagation.

    Attributes:
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of units in the output layer (1 for binary classification).
        dropout_rate (float): Dropout rate for regularization.
        features_group (List[List[int]]): List of feature indices for each group.
    """
    def __init__(self, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]], epochs: int = 1000, learning_rate: float = 0.01) -> None:
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.features_group = features_group
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.logger = TrainingLogger()
        
        # Initialize weights for each feature group
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for group in features_group:
            group_size: int = len(group)
            self.weights.append(np.random.randn(group_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
        
        self.W2 = np.random.randn(hidden_size * len(features_group), output_size)
        self.b2 = np.zeros((1, output_size))
        self._mlp: Optional[MLP] = None
        self.a1: Optional[List[np.ndarray]] = None  # Store intermediate activations
        
    def sigmoid(self, z: float or np.ndarray) -> float or np.ndarray: # type: ignore
        """
        Sigmoid activation function.

        Args:
            z (float or np.ndarray): The input value(s) to the sigmoid function.

        Returns:
            float or np.ndarray: The output value(s) after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z: float or np.ndarray) -> float or np.ndarray: # type: ignore
        """
        Calculate derivative of sigmoid function.

        Args:
            z (float or np.ndarray): The input value(s) to the sigmoid function.

        Returns:
            float or np.ndarray: Derivative of the sigmoid function.
        """
        return z * (1 - z)
    
    def dropout(self, layer_output: np.ndarray) -> np.ndarray:
        """
        Apply dropout regularization to the layer output.

        Args:
            layer_output (np.ndarray): The output of the layer.

        Returns:
            np.ndarray: The layer output after applying dropout regularization.
        """
        mask = (np.random.rand(*layer_output.shape) < self.dropout_rate) / self.dropout_rate 
        return layer_output * mask
    
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Calculate the forward pass of the neural network.

        Args:
            X (np.ndarray): The input data.
            training (bool): Whether the forward pass is performed during training or not.

        Returns:
            np.ndarray: The output of the neural network.
        """
        group_outputs = []
        self.a1 = []
        for idx, group in enumerate(self.features_group):
            group_x = X[:, group]
            z1 = np.dot(group_x, self.weights[idx]) + self.biases[idx]
            a1 = self.sigmoid(z1)
            if training:
                a1 = self.dropout(a1)
            group_outputs.append(a1)
            self.a1.append(a1)
        
        concatenated_outputs = np.concatenate(group_outputs, axis=1)
        
        # Final output layer
        z2 = np.dot(concatenated_outputs, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        return a2
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss between predicted and true values.

        Args:
            y_pred (np.ndarray): The predicted values.
            y_true (np.ndarray): The true values.

        Returns:
            float: The computed loss.
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid division by zero and log(0)
        loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, learning_rate: float) -> None:
        """
        Update the weights and biases of the neural network using backpropagation.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target labels.
            y_pred (np.ndarray): The predicted labels.
            learning_rate (float): The learning rate for updating the weights and biases.
        """
        m: int = y.shape[0]
        
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid division by zero
        
        # Compute gradients
        d_loss_a2: np.ndarray = -(y / y_pred) + ((1 - y) / (1 - y_pred))
        d_loss_z2: np.ndarray = d_loss_a2 * self.sigmoid_derivative(y_pred)
        
        d_loss_W2: np.ndarray = np.dot(np.concatenate(self.a1, axis=1).T, d_loss_z2) / m
        d_loss_b2: np.ndarray = np.sum(d_loss_z2, axis=0, keepdims=True) / m 
        
        d_loss_a1: np.ndarray = np.dot(d_loss_z2, self.W2.T)
        d_loss_z1: List[np.ndarray] = [d_loss_a1[:, i * self.hidden_size:(i + 1) * self.hidden_size] * self.sigmoid_derivative(self.a1[i])
                     for i in range(len(self.features_group))]
        
        d_loss_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        d_loss_biases: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        
        for idx, group in enumerate(self.features_group):
            group_x: np.ndarray = X[:, group]
            d_loss_weights[idx]: np.ndarray = np.dot(group_x.T, d_loss_z1[idx]) / m # type: ignore
            d_loss_biases[idx]: np.ndarray = np.sum(d_loss_z1[idx], axis=0, keepdims=True) / m # type: ignore
        
        # Update the weights and biases
        for idx in range(len(self.weights)):
            self.weights[idx] -= learning_rate * d_loss_weights[idx]
            self.biases[idx] -= learning_rate * d_loss_biases[idx]
        
        self.W2 -= learning_rate * d_loss_W2
        self.b2 -= learning_rate * d_loss_b2
        
    def calc_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the loss between predicted and true values.

        Args:
            y_pred (np.ndarray): The predicted values.
            y_true (np.ndarray): The true values.

        Returns:
            float: The calculated loss.
        """
        m: int = y_true.shape[0]
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid division by zero
        loss: float = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
        
    @ensure_valid_state
    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """
        Fits the MLP model to the given training data.

        Args:
            X (np.ndarray): The input features of shape (n_samples, n_features).
            y (np.ndarray): The target values of shape (n_samples,).

        Returns:
            MLP: The fitted MLP model.
        """
        self._mlp: MLP = self
        for epoch in range(self.epochs):
            y_pred: np.ndarray = self.forward(X, training=True)
            loss: float = self.compute_loss(y_pred, y)
            accuracy: float = np.mean(np.round(y_pred) == y)  # Calculate accuracy
            self.logger.log(epoch, loss, accuracy)  # Log the information
            self.logger.end_epoch()  # End the current epoch logging
        return self

    @ensure_valid_state
    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        y_pred: np.ndarray = self.forward(X, training=False)
        return np.round(y_pred)
    
    @ensure_valid_state
    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities of the target classes for the input data.

        Args:
            X (np.ndarray): The input data to be used for prediction.

        Returns:
            np.ndarray: The predicted probabilities of the target classes.
        """
        return self.forward(X, training=False)