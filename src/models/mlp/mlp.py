from typing import List
import numpy as np
from torch.nn import Module, Linear, ReLU, Dropout, Sigmoid, Sequential, BCELoss
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import wraps
import torch

# This state validation is code from Scikit-Longitudinal package
def ensure_valid_state(method):
    """
    Decorator function that ensures the model is in a valid state before executing the decorated method.

    Args:
        method (function): The method to be wrapped.

    Raises:
        ValueError: If the model has not been fitted yet and the wrapped method is `_predict` or `_predict_proba`.
        ValueError: If the feature groups have not been set and the wrapped method is `fit`.

    Returns:
        function: The wrapped method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if method.__name__ in ["_predict", "_predict_proba"] and self._mlp is None:
            raise ValueError("Model has not been fitted yet. Call the fit method first before calling predict or predict_proba.")
        
        if hasattr(self, "features_group") and (self.features_group is None or len(self.features_group) <= 1):
            raise ValueError("Feature groups have not been set. Call the setup_features_group method first before calling fit.")
        
        return method(self, *args, **kwargs)
    return wrapper

class MLPModule(Module):
    """
    Multi-layer Perceptron classifier for longitudinal data with skorch 

    Args:
        input_size (int): Number of input features
        hidden_sizes (List[int]): List of sizes for hidden layers
        output_size (int): Number of output units
        dropout_rate (float): Dropout rate for regularization
        features_group (List[List[int]]): List of feature indices for each group
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float, features_group: List[List[int]]):
        super(MLPModule, self).__init__()
        
        self.features_group = features_group
        layers: List[Module] = []
        in_size: int = sum([len(group) for group in features_group])
        
        for h_size in hidden_sizes:
            layers.append(Linear(in_size, h_size))
            layers.append(ReLU())
            layers.append(Dropout(dropout_rate))
            in_size = h_size

        layers.append(Linear(in_size, output_size))
        layers.append(Sigmoid())
        self.model: Module = Sequential(*layers)

    @property
    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        group_outputs = []
        for group in self.features_group:
            group_x = X[:, group]
            group_outputs.append(group_x)
        
        concatenated_outputs = torch.cat(group_outputs, dim=1)
        final_output = self.model(concatenated_outputs)
        return final_output

class BaseMLP(BaseEstimator, ClassifierMixin):
    """Base class for MLP models with temporal vector handling.

    This class provides a base implementation for MLP (Multi-Layer Perceptron) models.
    The goal of this class is to provide a common interface for all MLP models. So, all
    the derived classes should implement the `_fit`, `_predict`, and `_predict_proba` 
    methods. It inherits from `BaseEstimator` and `ClassifierMixin` classes.

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

class TemporalMLP(BaseMLP):
    """
    Wrapper class for skorch NeuralNetClassifier with MLPModule.

    Attributes:
        net (NeuralNetClassifier): skorch NeuralNetClassifier instance.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float, max_epochs: int, lr: float, features_group: List[List[int]]):
        self.input_size: int = input_size
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.output_size: int = output_size
        self.dropout_rate: float = dropout_rate
        self.max_epochs: int= max_epochs
        self.lr: float = lr
        self.features_group: List[List[int]] = features_group
        
        self.net: NeuralNetClassifier = NeuralNetClassifier(
            MLPModule,
            module__input_size=input_size,
            module__hidden_sizes=hidden_sizes,
            module__output_size=output_size,
            module__dropout_rate=dropout_rate,
            module_features_group=features_group,
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
        self.classes_: np.ndarray = np.unique(y) # Save the unique classes for later use
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

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the parameters of the model.

        Args:
            deep (bool, optional): If True, return the parameters of the model. Defaults to True.

        Returns:
            dict: The parameters of the model.
        """
        return super().get_params(deep)
    
    def set_params(self, **params) -> "TemporalMLP":
        """
        Set the parameters of the model.

        Args:
            **params: The parameters to be set.

        Returns:
            TemporalMLP: The model with the parameters set.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self