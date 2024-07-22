from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

DATASET_PATH = "./data/target_dataset.csv" # Path to the dataset
TARGET_WAVE = "class_target_w8" # Target Feature class

# Sample data
data: pd.DataFrame = pd.read_csv(DATASET_PATH)
features: List[str] = data.columns.tolist() 

# Split features into waves
waves: dict = {}
for feature in features:
    if feature.startswith('class_'):
        wave: str = feature.split('_')[-1]
    else:
        wave: str = feature.split('_')[-1][-2:]
    if wave not in waves:
        waves[wave]: List[str] = [] # type: ignore
    waves[wave].append(feature)

# Preprocess data, if not a float or int, replace with 0
for wave, features in waves.items():
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        data[feature] = data[feature].fillna(0)

# Filter waves dynamically
target_wave = TARGET_WAVE.split('_')[-1]
waves_to_exclude = [wave for wave in waves if wave != target_wave]

# Create a copy of the dataset
filtered_data = data.copy()
for wave in waves_to_exclude:
    class_vars = [feat for feat in waves[wave] if feat.startswith('class_')]
    filtered_data = filtered_data.drop(columns=class_vars, errors='ignore')

# Prepare the dataset for training
print(filtered_data.columns)

X = filtered_data.drop(columns=[TARGET_WAVE]).values
y = filtered_data[TARGET_WAVE].values

print(X.shape, y.shape)


# Split dataset - training, testing. Validation not implemented
train_size: int = int(0.8 * X.shape[0])
X_train: np.ndarray = X[:train_size]
X_test: np.ndarray = X[train_size:]
y_train: np.ndarray = y[:train_size]
y_test: np.ndarray = y[train_size:]

# Normalise data
scaler: MinMaxScaler = MinMaxScaler()
X_train: np.ndarray = scaler.fit_transform(X_train)
X_test: np.ndarray = scaler.transform(X_test)

# MLP implementation
"""_summary_
# Longitudinal Multi-layer Perceptron

Here I want to test a simple MLP model on longitudinal data. 
The model uses feedforward, with a single hidden layer, using a sigmoid activation function. The model is trained using backpropagation.
- The number of nueral units in the input layer is equal to the number of features in the dataset. 
- The number of units in the hidden layer is a hyperparameter.
- The number of units in the output layer is 1, as this is a binary classification problem. (Either patient is or is not diagnosed.)

Returns:
    _type_: MLP
"""
class MLP:
    def __init__(self: object, input_size: int, hidden_size: int, output_size: int, dropout_rate: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) #W1 is the weight matrix between the input layer and the hidden layer
        self.b1 = np.zeros((1, hidden_size)) #b1 is the bias vector for the hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) #W2 is the weight matrix between the hidden layer and the output layer
        self.b2 = np.zeros((1, output_size)) #b2 is the bias vector for the output layer
        
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
    
    def forward(self: object, X: np.ndarray) -> np.ndarray:
        """Calculates the forward pass of the neural network.

        Args:
            self (object): The neural network object.
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the neural network.
        """
        self.z1: np.ndarray = np.dot(X, self.W1) + self.b1
        self.a1: np.ndarray = self.sigmoid(self.z1) 
        
        self.z2: np.ndarray = np.dot(self.a1, self.W2) + self.b2
        self.a2: np.ndarray = self.sigmoid(self.z2)
        
        return self.a2
    
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
        
        # Compute gradients
        d_loss_a2: np.ndarray = -(y / y_pred) + ((1 - y) / (1 - y_pred))
        d_loss_z2: np.ndarray = d_loss_a2 * self.sigmoid_derivative(y_pred)
        
        d_loss_W2: np.ndarray = np.dot(self.a1.T, d_loss_z2) / m
        d_loss_b2: np.ndarray = np.sum(d_loss_z2, axis=0, keepdims=True) / m 
        
        d_loss_a1: np.ndarray = np.dot(d_loss_z2, self.W2.T)
        d_loss_z1: np.ndarray = d_loss_a1 * self.sigmoid_derivative(self.a1)
        
        d_loss_W1: np.ndarray = np.dot(X.T, d_loss_z1) / m
        d_loss_b1: np.ndarray = np.sum(d_loss_z1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= learning_rate * d_loss_W1
        self.b1 -= learning_rate * d_loss_b1
        self.W2 -= learning_rate * d_loss_W2
        self.b2 -= learning_rate * d_loss_b2
        
    def train(self: object, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> None:
            """Train the model using the given data.

            Args:
                self (object): The object instance.
                X (np.ndarray): The input data.
                y (np.ndarray): The target data.
                epochs (int): The number of training epochs.
                learning_rate (float): The learning rate for the optimizer.
            """
            for epoch in range(epochs):
                y_pred: np.ndarray = self.forward(X)
                loss: float = self.compute_loss(y_pred, y) 
                self.backward(X, y, y_pred, learning_rate)
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')
                
    def predict(self: object, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Args:
            self (object): The instance of the class.
            X (np.ndarray): The input data for prediction.

        Returns:
            np.ndarray: The predicted output.
        """
        y_pred: np.ndarray = self.forward(X)
        return np.round(y_pred)
    
# Model Parameters
input_size: int = X_train.shape[1]
hidden_size: int = 128
output_size: int = 1
epochs: int = 1000
learning_rate: float = 0.01
dropout_rate: float = 0.2

mlp: object = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, y_train.reshape(-1, 1), epochs, learning_rate)

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