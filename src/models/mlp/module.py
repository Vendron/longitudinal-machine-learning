from typing import List
from torch.nn import Module, Linear, ReLU, Dropout, Sigmoid, Sequential
from torch import Tensor, cat

class MLPModule(Module):
    """
    Multi-layer Perceptron classifier for longitudinal data with skorch 

    Args:
        input_size (int): Number of input features
        hidden_sizes (List[int]): List of sizes for hidden layers
        output_size (int): Number of output units
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float, features_group: List[List[int]]):
        super(MLPModule, self).__init__()
        
        self.features_group: List[List[int]] = features_group
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

    def forward(self, X) -> Tensor:
        """
        Pass forward the input tensor through the model.
        The input tensor is split into groups of features, 
        which are then passed through the model separately.
        The output of the model is then concatenated and passed through the model again
        to get the final output.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        grouped_features: List[Tensor] = []

        for group in self.features_group:
            group_x: Tensor = X[:, group]
            #group_output: Tensor = self.model(group_x)
            grouped_features.append(group_x)
            
        concatonated_features: Tensor = cat(grouped_features, dim=1)
        final_output: Tensor = self.model(concatonated_features)
        return final_output