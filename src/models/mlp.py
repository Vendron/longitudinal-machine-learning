from typing import List
from torch.nn import Module, Linear, ReLU, Dropout, Sigmoid, Sequential, BCELoss
import torch.optim as optim
from skorch import NeuralNetClassifier

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

class MLP:
    """
    Wrapper class for skorch NeuralNetClassifier with MLPModule.

    Attributes:
        net (NeuralNetClassifier): skorch NeuralNetClassifier instance.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float, max_epochs: int, lr: float):
        self.net = NeuralNetClassifier(
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

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.
        """
        self.net.fit(X, y)

    def predict(self, X):
        """
        Predict the output for the given input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.net.predict(X)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return self.net.predict_proba(X)