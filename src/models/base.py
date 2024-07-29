from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

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