from typing import List, Optional
import numpy as np
from torch.nn import Module, Linear, ReLU, Dropout, Sigmoid, Sequential, BCELoss
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import wraps

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
    """Base class for MLP models with temporal vector handling.

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


class MLPModule(Module):
    """
    Multi-layer Perceptron classifier for longitudinal data with skorch 

    Args:
        input_size (int): Number of input features
        hidden_sizes (List[int]): List of sizes for hidden layers
        output_size (int): Number of output units
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float):
        super(MLPModule, self).__init__()
        
        layers: List[Module] = []
        in_size: int = input_size
        
        for h_size in hidden_sizes:
            layers.append(Linear(in_size, h_size))
            layers.append(ReLU())
            layers.append(Dropout(dropout_rate))
            in_size = h_size

        layers.append(Linear(in_size, output_size))
        layers.append(Sigmoid())
        self.model: Module = Sequential(*layers)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(X)

class TemporalMLP(BaseMLP):
    """
    Wrapper class for skorch NeuralNetClassifier with MLPModule.

    Attributes:
        net (NeuralNetClassifier): skorch NeuralNetClassifier instance.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float, max_epochs: int, lr: float, features_group: List[List[int]]):
        self.features_group: List[List[int]] = features_group
        self.hidden_sizes: List[int] = hidden_sizes
        self.output_size: int = output_size
        self.dropout_rate: float = dropout_rate
        self.max_epochs: int= max_epochs
        self.lr: float = lr
        self.net: NeuralNetClassifier = NeuralNetClassifier(
            MLPModule,
            module__input_size=input_size,
            module__hidden_sizes=hidden_sizes,
            module__output_size=output_size,
            module__dropout_rate=dropout_rate,
            max_epochs=max_epochs,
            lr=lr,
            optimizer=optim.Adam,
            criterion=BCELoss,
            iterator_train__shuffle=True,
            verbose=1
        )

    @ensure_valid_state
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalMLP":
        """
        Fits the TemporalMLP model to the given training data.

        Args:
            X (np.ndarray): The input features of shape (n_samples, n_features).
            y (np.ndarray): The target values of shape (n_samples,).

        Returns:
            TemporalMLP: The fitted MLP model.
        """
        self._mlp: TemporalMLP = self
        self.net.fit(X, y)
        return self

    @ensure_valid_state
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return self.net.predict(X)
    
    @ensure_valid_state
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities of the target classes for the input data.

        Args:
            X (np.ndarray): The input data to be used for prediction.

        Returns:
            np.ndarray: The predicted probabilities of the target classes.
        """
        return self.net.predict_proba(X)