import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, classification_report, precision_recall_curve
)
from scikit_longitudinal.data_preparation import LongitudinalDataset
import os
from dotenv import load_dotenv

load_dotenv()

# Constants
DATASET_PATH = os.getenv("DATASET_PATH")
TARGET_WAVE = os.getenv("TARGET_WAVE")

def preprocess_data(data):
    """Preprocess the data."""
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    return data

def load_and_prepare_data(dataset_path, target_wave):
    """Load and prepare the dataset."""
    dataset = LongitudinalDataset(dataset_path)
    dataset.load_data_target_train_test_split(target_column=target_wave, random_state=42)
    dataset.setup_features_group("elsa")

    X_train = preprocess_data(dataset.X_train)
    X_test = preprocess_data(dataset.X_test)
    y_train = pd.to_numeric(dataset.y_train, errors='coerce').values
    y_test = pd.to_numeric(dataset.y_test, errors='coerce').values

    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

    return X_train, X_test, y_train, y_test, dataset.feature_groups()

def clean_feature_groups(feature_groups):
    """Clean and verify the feature groups."""
    cleaned_feature_groups = []
    for features in feature_groups:
        cleaned_features = [f for f in features if isinstance(f, str) and f != '-1']
        cleaned_feature_groups.append(cleaned_features)
    return cleaned_feature_groups

def reshape_data_for_lstm(X, feature_groups):
    """Reshape the data for LSTM."""
    feature_groups = clean_feature_groups(feature_groups)
    
    n_samples = X.shape[0]
    n_timesteps = len(feature_groups)
    max_features = max(len(features) for features in feature_groups)

    X_reshaped = np.zeros((n_samples, n_timesteps, max_features))
    for i, features in enumerate(feature_groups):
        try:
            indices = [X.columns.get_loc(f) for f in features]
            X_reshaped[:, i, :len(indices)] = X.iloc[:, indices]
        except KeyError as e:
            print(f"KeyError: {e}. This means one of the features is missing from the DataFrame columns.")
            print(f"Missing feature(s): {[f for f in features if f not in X.columns]}")
            print(f"All available columns: {list(X.columns)}")
            print(f"Feature groups: {feature_groups}")
            raise

    return X_reshaped

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.dropout(h_lstm[:, -1, :])
        out = self.fc(h_lstm)
        out = self.sigmoid(out)
        return out

def train_model(model, train_loader, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_batch = model(X_batch)
            y_pred.extend(y_pred_batch.cpu().numpy())

    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
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
    X_train, X_test, y_train, y_test, feature_groups = load_and_prepare_data(DATASET_PATH, TARGET_WAVE)
    
    X_train_reshaped = reshape_data_for_lstm(X_train, feature_groups)
    X_test_reshaped = reshape_data_for_lstm(X_test, feature_groups)
    
    train_dataset = TensorDataset(torch.tensor(X_train_reshaped, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_reshaped, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train_reshaped.shape[2]
    hidden_dim = 50
    output_dim = 1
    n_layers = 2
    n_epochs = 100
    
    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, n_epochs)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
