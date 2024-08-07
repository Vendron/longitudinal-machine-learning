import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import Module, LSTM, Dropout, Linear, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report

import os
from dotenv import load_dotenv

load_dotenv()

# Constants
DATASET_PATH = os.getenv("DATASET_PATH")
TARGET_WAVE = os.getenv("TARGET_WAVE")
TARGET_NAME: str = DATASET_PATH.split("/")[-1].split("_")[0] # e.g. 'dementia' from 'dementia_data.csv'

# Hyperparameters
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
BATCH_SIZE = 64
BIDIRECTIONAL = True
DROPOUT = 0.5
LSTM_UNITS = 100
DENSE_UNITS = 1
ACTIVATION = 'sigmoid'
LOSS = BCEWithLogitsLoss()
METRICS = ['accuracy']
VALIDATION_SPLIT = 0.1
PATIENCE = 10
NUM_LAYERS = 2
TEST_SIZE = 0.1

print(f"Dataset: {DATASET_PATH}")
print(f"Target class: {TARGET_WAVE}")

data = pd.read_csv(DATASET_PATH)
data.replace('?', np.nan, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)
# Remove class variables except the target for wave 8
class_vars_to_remove = [col for col in data.columns if f"class_{TARGET_NAME}_w" in col and TARGET_WAVE not in col]
data_copy = data.drop(columns=class_vars_to_remove)
print(f"Removed class variables: {class_vars_to_remove}")

# Separate features and target variable
X = data_copy.drop(columns=[TARGET_WAVE])
y = data_copy[TARGET_WAVE]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Identify features by wave
column_names = X.columns
wave_identifiers = sorted(set(col.split('_')[-1] for col in column_names if col != 'sex' and col != 'indager_wave8' and col != 'dheas_wave4' and col != 'apoe_wave2'))
non_longitudinal_features = ['sex', 'indager_wave8', 'dheas_wave4', 'apoe_wave2']

print(f"Wave identifiers: {wave_identifiers}")
print(non_longitudinal_features)

# Group features by waves
features_by_wave = {wave: [] for wave in wave_identifiers}
for col in column_names:
    if col not in non_longitudinal_features:
        wave = col.split('_')[-1]
        features_by_wave[wave].append(col)
        print(wave, col)
        
# Prepare data for RNN
n_samples = X_scaled.shape[0]
n_timesteps = len(wave_identifiers)
n_features_per_wave = {wave: len(features) for wave, features in features_by_wave.items()}
max_features = max(n_features_per_wave.values())

print(f"Number of samples: {n_samples}")
print(f"Number of waves: {n_timesteps}")
print(f"Wave identifiers: {wave_identifiers}")
print(f"Number of features per wave: {n_features_per_wave}")

# Reshape data without non-longitudinal features
X_reshaped = np.zeros((n_samples, n_timesteps, max_features))
for i, wave in enumerate(wave_identifiers):
    wave_features = features_by_wave[wave]
    indices = [column_names.get_loc(f) for f in wave_features]
    X_reshaped[:, i, :len(indices)] = X_scaled[:, indices]
print(f"X reshaped shape: {X_reshaped.shape}")

# Add non-longitudinal features to every timestep
for feature in non_longitudinal_features:
    feature_index = column_names.get_loc(feature)
    expanded_features = np.repeat(X_scaled[:, feature_index][:, np.newaxis], n_timesteps, axis=1)
    X_reshaped[:, :, -len(non_longitudinal_features) + non_longitudinal_features.index(feature)] = expanded_features

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=TEST_SIZE, random_state=42)

# PyTorch Dataset and DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print(f"Wave identifiers: {wave_identifiers}")
print(f"Number of features per wave: {n_features_per_wave}")

class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=False, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = Linear(hidden_dim * direction_factor, output_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.dropout(h_lstm[:, -1, :])  # Get the output of the last LSTM cell
        out = self.fc(h_lstm)
        out = self.sigmoid(out)
        return out

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
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
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

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[2]
model = LSTMModel(input_dim, LSTM_UNITS, DENSE_UNITS, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

train_model(model, train_loader, LOSS, optimizer, MAX_EPOCHS, device)
evaluate_model(model, test_loader, device)