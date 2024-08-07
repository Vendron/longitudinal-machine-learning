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

def load_and_preprocess_data(dataset_path, target_class):
    """Load and preprocess the dataset."""
    data = pd.read_csv(dataset_path)
    data.replace('?', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    
    class_vars_to_remove = [col for col in data.columns if f"class_{TARGET_NAME}_w" in col and target_class not in col]
    data_copy = data.drop(columns=class_vars_to_remove)
    print(f"Removed class variables: {class_vars_to_remove}")
    
    X = data_copy.drop(columns=[target_class])
    y = data_copy[target_class]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

def normalize_data(X):
    """Normalize the data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def group_features_by_waves(column_names):
    """Group features by their respective waves."""
    wave_identifiers = sorted(set(col.split('_')[-1] for col in column_names if col not in ['sex', 'indager_wave8', 'dheas_wave4', 'apoe_wave2']))
    features_by_wave = {wave: [] for wave in wave_identifiers}
    
    for col in column_names:
        if col not in ['sex', 'indager_wave8', 'dheas_wave4', 'apoe_wave2']:
            wave = col.split('_')[-1]
            features_by_wave[wave].append(col)
    
    return features_by_wave, wave_identifiers

def reshape_data(X_scaled, features_by_wave, wave_identifiers, column_names):
    """Reshape data for LSTM."""
    n_samples = X_scaled.shape[0]
    n_timesteps = len(wave_identifiers)
    max_features = max(len(features) for features in features_by_wave.values())
    
    X_reshaped = np.zeros((n_samples, n_timesteps, max_features))
    for i, wave in enumerate(wave_identifiers):
        wave_features = features_by_wave[wave]
        indices = [column_names.get_loc(f) for f in wave_features]
        X_reshaped[:, i, :len(indices)] = X_scaled[:, indices]
    
    return X_reshaped

def prepare_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    """Prepare PyTorch DataLoader for training and testing."""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

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

def main():
    """Main function to run the workflow."""
    X, y = load_and_preprocess_data(DATASET_PATH, TARGET_WAVE)
    X_scaled = normalize_data(X)
    features_by_wave, wave_identifiers = group_features_by_waves(X.columns)
    X_reshaped = reshape_data(X_scaled, features_by_wave, wave_identifiers, X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=TEST_SIZE, random_state=42)
    train_loader, test_loader = prepare_dataloaders(X_train, X_test, y_train, y_test, BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[2]

    model = LSTMModel(input_dim, LSTM_UNITS, DENSE_UNITS, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, LOSS, optimizer, MAX_EPOCHS, device)
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()