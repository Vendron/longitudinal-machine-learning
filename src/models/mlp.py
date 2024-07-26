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
    def __init__(self, hidden_layer_sizes: List[int], output_size: int, dropout_rate: float, features_group: List[List[int]], epochs: int = 1000, learning_rate: float = 0.01) -> None:
        self.hidden_layer_sizes: List[int] = hidden_layer_sizes
        self.output_size: int = output_size
        self.dropout_rate: float = dropout_rate
        self.features_group: List[List[int]] = features_group
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.logger: TrainingLogger = TrainingLogger()
        
        # Initialize weights for each feature group
        self.input_weights: List[np.ndarray] = []
        self.input_biases: List[np.ndarray] = []
        for group in features_group:
            group_size: int = len(group)
            self.input_weights.append(np.random.randn(group_size, hidden_layer_sizes[0]))
            self.input_biases.append(np.zeros((1, hidden_layer_sizes[0])))
        
        # Initialize weights for hidden layers
        self.hidden_weights: List[np.ndarray] = []
        self.hidden_biases: List[np.ndarray] = []
        input_size: int = hidden_layer_sizes[0] * len(features_group)
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_weights.append(np.random.randn(input_size, hidden_layer_sizes[i + 1]))
            self.hidden_biases.append(np.zeros((1, hidden_layer_sizes[i + 1])))
            input_size: int = hidden_layer_sizes[i + 1]
        
        # Initialize weights for output layer
        self.output_weights: np.ndarray = np.random.randn(input_size, output_size)
        self.output_biases: np.ndarray = np.zeros((1, output_size))
        self._mlp: Optional[MLP] = None
        self.a_hidden: Optional[List[np.ndarray]] = None  # Store intermediate activations
    
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
        mask: np.ndarray = (np.random.rand(*layer_output.shape) < self.dropout_rate) / self.dropout_rate 
        return layer_output * mask
    
    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
            """Calculate the forward propagation through the MLP model.

            Args:
                X (np.ndarray): Input data of shape (batch_size, num_features).
                training (bool, optional): Flag indicating whether the model is in training mode or not.
                    Defaults to False.

            Returns:
                np.ndarray: Output of the MLP model, of shape (batch_size, num_classes).
            """
            group_outputs: List[np.ndarray] = []
            self.a_hidden: List[np.ndarray] = []
            for idx, group in enumerate(self.features_group):
                group_x = X[:, group]
                z1: np.ndarray = np.dot(group_x, self.input_weights[idx]) + self.input_biases[idx]
                a1: np.ndarray = self.sigmoid(z1)
                if training:
                    a1: np.ndarray = self.dropout(a1)
                group_outputs.append(a1)
            
            concatenated_outputs: np.ndarray = np.concatenate(group_outputs, axis=1)
            self.a_hidden.append(concatenated_outputs)
            
            for i in range(len(self.hidden_weights)):
                z_hidden: np.ndarray = np.dot(self.a_hidden[-1], self.hidden_weights[i]) + self.hidden_biases[i]
                a_hidden: np.ndarray = self.sigmoid(z_hidden)
                if training:
                    a_hidden: np.ndarray = self.dropout(a_hidden)
                self.a_hidden.append(a_hidden)
            
            z_output: np.ndarray = np.dot(self.a_hidden[-1], self.output_weights) + self.output_biases
            a_output: np.ndarray = self.sigmoid(z_output)
            
            return a_output
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Performs the backward pass of the MLP model to compute gradients and update weights and biases.

        Args:
            X (np.ndarray): The input data of shape (m, n), where m is the number of samples and n is the number of features.
            y (np.ndarray): The target labels of shape (m,).
            y_pred (np.ndarray): The predicted labels of shape (m,).
            learning_rate (float): The learning rate for updating the weights and biases.

        Returns:
            None
        """
        num_samples: int = y_true.shape[0]
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid division by zero and log(0)
        loss: float = -(1/num_samples) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
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
        num_samples: int = y.shape[0]
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid division by zero
        
        # Compute gradients
        d_loss_a2: np.ndarray = -(y / y_pred) + ((1 - y) / (1 - y_pred))
        d_loss_z2: np.ndarray = d_loss_a2 * self.sigmoid_derivative(y_pred)
        
        d_loss_output_weights: np.ndarray = np.dot(self.a_hidden[-1].T, d_loss_z2) / num_samples
        d_loss_output_biases: np.ndarray = np.sum(d_loss_z2, axis=0, keepdims=True) / num_samples
        
        d_loss_a_hidden: np.ndarray = np.dot(d_loss_z2, self.output_weights.T)
        
        d_loss_hidden_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.hidden_weights]
        d_loss_hidden_biases: List[np.ndarray] = [np.zeros_like(b) for b in self.hidden_biases]
        
        for i in range(len(self.hidden_weights) - 1, -1, -1):
            d_loss_z_hidden = d_loss_a_hidden * self.sigmoid_derivative(self.a_hidden[i + 1])
            d_loss_hidden_weights[i] = np.dot(self.a_hidden[i].T, d_loss_z_hidden) / num_samples
            d_loss_hidden_biases[i] = np.sum(d_loss_z_hidden, axis=0, keepdims=True) / num_samples
            d_loss_a_hidden = np.dot(d_loss_z_hidden, self.hidden_weights[i].T)
        
        d_loss_input_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.input_weights]
        d_loss_input_biases: List[np.ndarray] = [np.zeros_like(b) for b in self.input_biases]
        
        for idx, group in enumerate(self.features_group):
            group_x: np.ndarray = X[:, group]
            d_loss_input_weights[idx]: np.ndarray = np.dot(group_x.T, d_loss_a_hidden[:, idx * self.hidden_layer_sizes[0]:(idx + 1) * self.hidden_layer_sizes[0]]) / num_samples # type: ignore
            d_loss_input_biases[idx]: np.ndarray = np.sum(d_loss_a_hidden[:, idx * self.hidden_layer_sizes[0]:(idx + 1) * self.hidden_layer_sizes[0]], axis=0, keepdims=True) / num_samples # type: ignore
        
        # Update weights and biases
        for idx in range(len(self.input_weights)):
            self.input_weights[idx] -= learning_rate * d_loss_input_weights[idx]
            self.input_biases[idx] -= learning_rate * d_loss_input_biases[idx]
        
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= learning_rate * d_loss_hidden_weights[i]
            self.hidden_biases[i] -= learning_rate * d_loss_hidden_biases[i]
        
        self.output_weights -= learning_rate * d_loss_output_weights
        self.output_biases -= learning_rate * d_loss_output_biases
        
    def calc_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the loss between predicted and true values.

        Args:
            y_pred (np.ndarray): The predicted values.
            y_true (np.ndarray): The true values.

        Returns:
            float: The calculated loss.
        """
        num_samples: int = y_true.shape[0]
        y_pred: np.ndarray = np.clip(y_pred, 1e-10, 1 - 1e-10) # Avoid division by zero
        loss: float = -(1/num_samples) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
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
            accuracy: float = np.mean(np.round(y_pred) == y) # Calculate accuracy
            self.logger.log(epoch, loss, accuracy) # Log the current epoch, loss, and accuracy
            self.backward(X, y, y_pred, self.learning_rate) # Perform backpropagation
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