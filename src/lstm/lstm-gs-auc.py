import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import Module, LSTM, Dropout, Linear, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report
from itertools import product
import glob
import os
from sklearn.model_selection import KFold
import torch.multiprocessing as mp

# Constants
DATA_DIRECTORY = "../data"
DATASET_PATTERN = "*_dataset.csv"
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
BATCH_SIZE = 32
BIDIRECTIONAL = True
DROPOUT = 0.5
LSTM_UNITS = 100
DENSE_UNITS = 1
ACTIVATION = 'sigmoid'
LOSS = BCEWithLogitsLoss()
VALIDATION_SPLIT = 0.1
PATIENCE = 10
NUM_LAYERS = 2
TEST_SIZE = 0.1
CONCURRENT_PROCESSES = 8  # Start with 4 processes

def log_to_file(log_file, message):
    """Utility function to log a message to a file."""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=False, dropout=0.0):
        super(LSTMModel, self).__init__()
        if n_layers > 1:
            self.lstm = LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            self.lstm = LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        
        self.dropout = Dropout(dropout) if dropout > 0 else None
        direction_factor = 2 if bidirectional else 1
        self.fc = Linear(hidden_dim * direction_factor, output_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        if self.dropout is not None:
            h_lstm = self.dropout(h_lstm[:, -1, :])
        else:
            h_lstm = h_lstm[:, -1, :]
        out = self.fc(h_lstm)
        out = self.sigmoid(out)
        return out

def train_validate_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    """Train the model and evaluate on validation set."""
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
    
    # Evaluate on validation set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_batch = model(X_batch)
            y_pred.extend(y_pred_batch.cpu().numpy())
    
    y_pred = np.array(y_pred).squeeze()
    roc_auc = roc_auc_score(y_true, y_pred)
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
    return roc_auc, conf_matrix

def grid_search(log_file_name, input_dim, train_data, val_data, hyperparameter_grid, device, n_epochs=10):
    best_hyperparameters = None
    best_auc = float('-inf')
    best_conf_matrix = None

    for lr, dropout, n_layers, lstm_units in hyperparameter_grid:
        model = LSTMModel(input_dim, lstm_units, DENSE_UNITS, n_layers, BIDIRECTIONAL, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        auc, conf_matrix = train_validate_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
        
        if auc > best_auc:
            best_auc = auc
            best_conf_matrix = conf_matrix
            best_hyperparameters = (lr, dropout, n_layers, lstm_units)
            log_to_file(log_file_name, f"Found new best hyperparameters: {best_hyperparameters}")

    return best_hyperparameters, best_auc, best_conf_matrix

def train_model(model, train_loader, criterion, optimizer, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader, device, log_file_name):
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
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall_curve, precision_curve)
    
    report = classification_report(y_true, y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)

    log_to_file(log_file_name, f'Test Precision: {precision:.4f}')
    log_to_file(log_file_name, f'Test Recall: {recall:.4f}')
    log_to_file(log_file_name, f'Test F1 Score: {f1:.4f}')
    log_to_file(log_file_name, f'Test ROC-AUC Score: {roc_auc:.4f}')
    log_to_file(log_file_name, f'Test AUPRC Score: {auprc:.4f}')
    log_to_file(log_file_name, f'Confusion Matrix:\n{conf_matrix}')
    log_to_file(log_file_name, f'Classification Report:\n{report}')


    return precision, recall, f1, roc_auc, auprc, report, conf_matrix

def process_dataset(dataset_path, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    TARGET_NAME = os.path.basename(dataset_path).split("_")[0]
    TARGET_CLASS = "class_" + TARGET_NAME + "_w8"
    
    log_file_name = f"lstm_{TARGET_NAME}_gs_auc_log.txt"
    
    log_to_file(log_file_name, f"Dataset: {dataset_path}")
    log_to_file(log_file_name, f"Target: {TARGET_NAME}")
    log_to_file(log_file_name, f"Target class: {TARGET_CLASS}")
    
    data = pd.read_csv(dataset_path)
    data.replace('?', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)

    target_wave_suffix = TARGET_CLASS.split('_')[-1]
    target_wave_number = int(target_wave_suffix[1:])

    # Remove class variables except the target wave
    class_vars_to_remove = [col for col in data.columns if col.startswith("class_") and TARGET_CLASS not in col]
    features_to_remove = [col for col in data.columns if any(col.endswith(f'w{wave}') for wave in range(target_wave_number + 1, 9))]

    columns_to_remove = class_vars_to_remove + features_to_remove
    data_copy = data.drop(columns=columns_to_remove)

    # Separate features and target variable
    X = data_copy.drop(columns=[TARGET_CLASS])
    y = data_copy[TARGET_CLASS]

    # Convert to PyTorch tensors and move them to GPU
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

    # Normalize the data on the GPU
    X_tensor = (X_tensor - X_tensor.mean(dim=0)) / X_tensor.std(dim=0)

    # Identify features by wave
    column_names = X.columns
    wave_identifiers = sorted(set(col.split('_')[-1] for col in column_names if col != 'sex' and col != 'indager_wave8' and col != 'dheas_wave4' and col != 'apoe_wave2'))
    non_longitudinal_features = [col for col in column_names if not col.endswith('w1') and not col.endswith('w2') and not col.endswith('w3') and not col.endswith('w4') and not col.endswith('w5') and not col.endswith('w6') and not col.endswith('w7') and not col.endswith('w8')]

    features_by_wave = {wave: [] for wave in wave_identifiers}
    for col in column_names:
        if col not in non_longitudinal_features:
            wave = col.split('_')[-1]
            features_by_wave[wave].append(col)

    n_samples = X_tensor.shape[0]
    n_timesteps = len(wave_identifiers)
    n_features_per_wave = {wave: len(features) for wave, features in features_by_wave.items()}
    max_features = max(n_features_per_wave.values())

    X_reshaped = torch.zeros((n_samples, n_timesteps, max_features), device=device)
    for i, wave in enumerate(wave_identifiers):
        wave_features = features_by_wave[wave]
        indices = [column_names.get_loc(f) for f in wave_features]
        X_reshaped[:, i, :len(indices)] = X_tensor[:, indices]
        
    print("Shape of X_reshaped:", X_reshaped.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_tensor, test_size=TEST_SIZE, random_state=42)

    # PyTorch Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    input_dim = X_train.shape[2]
    hidden_dim = LSTM_UNITS
    output_dim = DENSE_UNITS
    n_layers = NUM_LAYERS

    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers, BIDIRECTIONAL, DROPOUT).to(device)
    criterion = LOSS
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.0001]
    dropout_rates = [0.2, 0.5, 0.3]
    n_layers_options = [1, 2, 3]
    lstm_units_options = [50, 100, 150]

    # All combinations of hyperparameters
    hyperparameter_grid = list(product(learning_rates, dropout_rates, n_layers_options, lstm_units_options))

    # K-Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_results_per_fold = []

    for fold, (train_index, test_index) in enumerate(kf.split(X_reshaped.cpu().numpy())):
        log_to_file(log_file_name, f"Fold {fold + 1} at {time.ctime()}")
        print(f"{TARGET_NAME} Fold {fold + 1} at {time.ctime()}") 
        X_train_fold, X_test_fold = X_reshaped[train_index], X_reshaped[test_index]
        y_train_fold, y_test_fold = y_tensor[train_index], y_tensor[test_index]
        
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_fold.cpu().numpy(), y_train_fold.cpu().numpy(), test_size=VALIDATION_SPLIT, random_state=42)

        X_train_sub, X_val_sub, y_train_sub, y_val_sub = torch.tensor(X_train_sub, dtype=torch.float32).to(device), torch.tensor(X_val_sub, dtype=torch.float32).to(device), torch.tensor(y_train_sub, dtype=torch.float32).to(device), torch.tensor(y_val_sub, dtype=torch.float32).to(device)
        
        train_data_sub = TensorDataset(X_train_sub, y_train_sub)
        val_data_sub = TensorDataset(X_val_sub, y_val_sub)
        
        best_hyperparameters, best_auc, best_conf_matrix = grid_search(log_file_name, input_dim, train_data_sub, val_data_sub, hyperparameter_grid, device, n_epochs=MAX_EPOCHS)
        
        print(f"Best Hyperparameters for Fold {fold + 1}: {best_hyperparameters}")        
        log_to_file(log_file_name, f"Best Hyperparameters for Fold {fold + 1}: {best_hyperparameters}")
        
        best_results_per_fold.append([best_hyperparameters, best_auc, best_conf_matrix])
        
        lr, dropout, n_layers, lstm_units = best_hyperparameters
        print(f"Training on full training set with best hyperparameters: {best_hyperparameters}")
        model = LSTMModel(input_dim, lstm_units, output_dim, n_layers, BIDIRECTIONAL, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        
        train_data_final = TensorDataset(X_train_fold, y_train_fold)
        test_data_final = TensorDataset(X_test_fold, y_test_fold)
        
        train_loader_final = DataLoader(train_data_final, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader_final = DataLoader(test_data_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        train_model(model, train_loader_final, criterion, optimizer, MAX_EPOCHS, device)
        evaluate_model(model, test_loader_final, device, log_file_name)


if __name__ == "__main__":
    dataset_paths = glob.glob(os.path.join(DATA_DIRECTORY, DATASET_PATTERN))
    num_gpus = torch.cuda.device_count()
    
    print(torch.cuda.is_available())

    num_processes = min(len(dataset_paths), CONCURRENT_PROCESSES)
    print(f"Found {len(dataset_paths)} datasets and {num_gpus} GPUs. Using {num_processes} processes.")
    
    mp.set_start_method('spawn') 

    with mp.Pool(processes=num_processes) as pool:
        print("Starting parallel processing...")
        print(f"Using {num_processes} processes.")
        pool.starmap(process_dataset, [(path, 0) for path in dataset_paths])

