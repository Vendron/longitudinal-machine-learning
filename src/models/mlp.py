import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from typing import List

class TemporalMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]]):
        """
        Initializes the TemporalMLP model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of units in the hidden layer.
            output_size (int): Number of units in the output layer.
            dropout_rate (float): Dropout rate for regularization.
            features_group (List[List[int]]): List of feature groups.
        """
        super(TemporalMLP, self).__init__()
        self.features_group = features_group
        self.hidden_size = hidden_size
        self.group_layers = nn.ModuleList()
        
        for group in features_group:
            group_size = len(group)
            self.group_layers.append(nn.Linear(group_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * len(features_group), output_size)
    
    def forward(self, X):
        """
        Defines the forward pass of the TemporalMLP model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        group_outputs = []
        for idx, group in enumerate(self.features_group):
            group_x = X[:, group]
            group_output = F.sigmoid(self.group_layers[idx](group_x))
            group_outputs.append(group_output)
        
        concatenated_outputs = torch.cat(group_outputs, dim=1)
        concatenated_outputs = self.dropout(concatenated_outputs)
        output = torch.sigmoid(self.fc(concatenated_outputs))
        
        return output

def build_skorch_model(input_size: int, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]]) -> NeuralNetClassifier:
    """
    Builds a NeuralNetClassifier using the TemporalMLP model.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of units in the output layer.
        dropout_rate (float): Dropout rate for regularization.
        features_group (List[List[int]]): List of feature groups.

    Returns:
        NeuralNetClassifier: The built NeuralNetClassifier.
    """
    net = NeuralNetClassifier(
        module=TemporalMLP,
        module__input_size=input_size,
        module__hidden_size=hidden_size,
        module__output_size=output_size,
        module__dropout_rate=dropout_rate,
        module__features_group=features_group,
        max_epochs=1000,
        lr=0.01,
        iterator_train__shuffle=True,
        train_split=None,  # No validation split
        verbose=1
    )
    return net