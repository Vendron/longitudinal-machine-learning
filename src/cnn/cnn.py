import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import Module, Conv1d, MaxPool1d, Linear, Dropout, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report

# Constants
DATASET_PATH = "../../data/<>.csv"
TARGET_NAME = DATASET_PATH.split("/")[-1].split("_")[0]
TARGET_CLASS = "class_" + TARGET_NAME + "_w8"

LEARNING_RATE = 0.001 # The learning rate for the optimizer 
MAX_EPOCHS = 100 # The maximum number of epochs to train the model
BATCH_SIZE = 32 # Number of samples per batch
DROPOUT = 0.5 # The dropout rate
CNN_CHANNELS = 64 # Number of output channels of the convolutional layer
KERNEL_SIZE = 3 # The size of the kernel in the convolutional layer
POOL_SIZE = 2 # The size of the pooling layer
DENSE_UNITS = 1 # Number of units in the dense layer
ACTIVATION = 'sigmoid'
LOSS = BCEWithLogitsLoss()
METRICS = ['accuracy']
VALIDATION_SPLIT = 0.1 # The percentage of the training set to use as validation
PATIENCE = 10 # The number of epochs to wait before early stopping if no improvement is made
NUM_LAYERS = 2 # The number of layers in the model
TEST_SIZE = 0.1 # The percentage of the dataset to use as test set

print(f"Dataset: {DATASET_PATH}")
print(f"Target: {TARGET_NAME}")
print(f"Target class: {TARGET_CLASS}")

data = pd.read_csv(DATASET_PATH)
data.replace('?', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(0, inplace=True)

# Extract features and target variables
target_wave_suffix = TARGET_CLASS.split('_')[-1]
target_wave_number = int(target_wave_suffix[1:])
class_vars_to_remove = [col for col in data.columns if col.startswith("class_") and TARGET_CLASS not in col]
features_to_remove = [col for col in data.columns if any(col.endswith(f'w{wave}') for wave in range(target_wave_number + 1, 9))]

columns_to_remove = class_vars_to_remove + features_to_remove
data_copy = data.drop(columns=columns_to_remove)

X = data_copy.drop(columns=[TARGET_CLASS])
y = data_copy[TARGET_CLASS]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Identify features by wave
column_names = X.columns
wave_identifiers = sorted(set(col.split('_')[-1] for col in column_names if col != 'sex' and col != 'indager_wave8' and col != 'dheas_wave4' and col != 'apoe_wave2'))
non_longitudinal_features = [col for col in column_names if not col.endswith('w1') and not col.endswith('w2') and not col.endswith('w3') and not col.endswith('w4') and not col.endswith('w5') and not col.endswith('w6') and not col.endswith('w7') and not col.endswith('w8')]

# Group features by waves
features_by_wave = {wave: [] for wave in wave_identifiers}
for col in column_names:
    if col not in non_longitudinal_features:
        wave = col.split('_')[-1]
        features_by_wave[wave].append(col)

# Prepare data
n_samples = X_scaled.shape[0]
n_timesteps = len(wave_identifiers)
n_features_per_wave = {wave: len(features) for wave, features in features_by_wave.items()}
max_features = max(n_features_per_wave.values())

# Reshape data without non-longitudinal features
X_reshaped = np.zeros((n_samples, n_timesteps, max_features))
for i, wave in enumerate(wave_identifiers):
    wave_features = features_by_wave[wave]
    indices = [column_names.get_loc(f) for f in wave_features]
    X_reshaped[:, i, :len(indices)] = X_scaled[:, indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=TEST_SIZE, random_state=42)

# PyTorch Dataset and DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the CNN model
class CNNModel(Module):
    def __init__(self, input_dim, num_channels, kernel_size, pool_size, dropout=0.0):
        super(CNNModel, self).__init__()
        self.conv1 = Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=kernel_size, padding=1)
        self.pool = MaxPool1d(pool_size)
        self.dropout = Dropout(dropout)
        self.fc = Linear(num_channels * (n_timesteps // pool_size), DENSE_UNITS)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change the dimension to (batch_size, input_dim, n_timesteps) for Conv1d
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        if self.dropout is not None:
            x = self.dropout(x)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[2]
model = CNNModel(input_dim=input_dim, num_channels=CNN_CHANNELS, kernel_size=KERNEL_SIZE, pool_size=POOL_SIZE, dropout=DROPOUT).to(device)
criterion = LOSS
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train_model(model, train_loader, criterion, optimizer, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad(): 
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_batch = model(X_batch)
            y_pred.extend(y_pred_batch.cpu().numpy())

    y_pred = np.array(y_pred).squeeze()
    
    # Convert predictions to binary values
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics using binary predictions
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # ROC-AUC and Precision-Recall AUC use the continuous y_pred
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall_curve, precision_curve)
    
    report = classification_report(y_true, y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)

    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC-AUC Score: {roc_auc:.4f}')
    print(f'Test AUPRC Score: {auprc:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{report}')
    return y_true, y_pred

# # Training the model
train_model(model, train_loader, criterion, optimizer, 100, device)
evaluate_model(model, test_loader, device)