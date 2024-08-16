import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn import Module, LSTM, Dropout, Linear, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report
import os
from dotenv import load_dotenv
from itertools import product
from sklearn.utils.class_weight import compute_class_weight

load_dotenv()

# Constants
DATASET_PATH = os.getenv("DATASET_PATH")
TARGET_WAVE = os.getenv("TARGET_WAVE")
TARGET_NAME = DATASET_PATH.split("/")[-1].split("_")[0]

# Hyperparameters
KFOLDS = 10
LOSS = BCEWithLogitsLoss()
BIDIRECTIONAL = True
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
DROPOUT = 0.5
LSTM_UNITS = 100
NUM_LAYERS = 2
BATCH_SIZE = 64
DENSE_UNITS = 1

# Define the hyperparameter grid for grid search
HYPERPARAMETER_GRID = {
    'learning_rate': [0.001, 0.01],
    'lstm_units': [50, 100],
    'dropout': [0.2, 0.5],
    'num_layers': [1, 2],
}

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
    """Normalize the data using MinMaxScaler."""
    scaler = MinMaxScaler()
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

def prepare_dataloaders(X_train, y_train, batch_size):
    """Prepare PyTorch DataLoader for training and testing."""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=False, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0)
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

def train_model(model, train_loader, criterion, optimizer, scheduler, n_epochs, device, early_stopping_patience=10):
    """Train the LSTM model."""
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
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
        
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping")
                break

    # Evaluate on validation set
    model.eval()
    y_true, y_pred = [], []
    
    y_pred = np.array(y_pred).squeeze()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
    return f1, conf_matrix, recall, precision


def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate the performance of a given model on the test data.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        X_test (numpy.ndarray): The input features of the test data.
        y_test (numpy.ndarray): The target labels of the test data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation results, including precision, recall, F1 score,
              ROC AUC score, AUPRC score, classification report, and confusion matrix.
    """
    model.eval()
    y_true = []
    y_pred = []
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_batch = model(X_batch)
            y_pred.extend(y_pred_batch.cpu().numpy())

    y_pred = np.array(y_pred).squeeze()
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=1)
    recall = recall_score(y_true, y_pred_binary, zero_division=1)
    f1 = f1_score(y_true, y_pred_binary, zero_division=1)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall_curve, precision_curve)
    report = classification_report(y_true, y_pred_binary, zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'report': report,
        'conf_matrix': conf_matrix
    }
    
    return results

def grid_search(X, y, param_grid, k_folds):
    """Perform grid search for hyperparameter tuning."""
    best_f1 = float('-inf')
    best_conf_matrix = None
    best_preicison = None
    best_recall = None
    best_params = None
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    param_list = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())
    
    for params in param_list:
        param_dict = dict(zip(param_keys, params))
        print(f"Testing parameters: {param_dict}")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f'Fold {fold+1}/{k_folds}')
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.values[train_idx], y.values[test_idx]
            
            train_loader = prepare_dataloaders(X_train, y_train, BATCH_SIZE)
            
            model = LSTMModel(
                input_dim=X_train.shape[2],
                hidden_dim=param_dict['lstm_units'],
                output_dim=DENSE_UNITS,
                n_layers=param_dict['num_layers'],
                bidirectional=BIDIRECTIONAL,
                dropout=param_dict['dropout']
            ).to(device)
            
            optimizer = Adam(model.parameters(), lr=param_dict['learning_rate'])
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
            
            # Pass the device argument to train_model
            train_model(model, train_loader, LOSS, optimizer, scheduler, MAX_EPOCHS, device)
            
            results = evaluate_model(model, X_test, y_test, device)
            fold_results.append(results)

        avg_score = np.mean([fold_result['f1'] for fold_result in fold_results])
        all_results.append((param_dict, fold_results))
        
        if avg_score > best_f1:
            best_f1 = avg_score
            best_params = param_dict

    return best_params, best_f1, all_results

def save_results_to_file(results, filename):
    """Save the results to a file."""
    with open(filename, 'w') as f:
        for param_dict, fold_results in results:
            f.write(f"Parameters: {param_dict}\n")
            for i, result in enumerate(fold_results):
                f.write(f"Fold {i+1} Results:\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"F1 Score: {result['f1']:.4f}\n")
                f.write(f"ROC-AUC Score: {result['roc_auc']:.4f}\n")
                f.write(f"AUPRC Score: {result['auprc']:.4f}\n")
                f.write(f"Classification Report:\n{result['report']}\n")
                f.write(f"Confusion Matrix:\n{result['conf_matrix']}\n\n")
            
            avg_precision = np.mean([result['precision'] for result in fold_results])
            avg_recall = np.mean([result['recall'] for result in fold_results])
            avg_f1 = np.mean([result['f1'] for result in fold_results])
            avg_roc_auc = np.mean([result['roc_auc'] for result in fold_results])
            avg_auprc = np.mean([result['auprc'] for result in fold_results])
            
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average F1 Score: {avg_f1:.4f}\n")
            f.write(f"Average ROC-AUC Score: {avg_roc_auc:.4f}\n")
            f.write(f"Average AUPRC Score: {avg_auprc:.4f}\n")
            f.write("="*50 + "\n")

def main():
    """Main function to run the workflow."""
    X, y = load_and_preprocess_data(DATASET_PATH, TARGET_WAVE)
    X_scaled = normalize_data(X)
    features_by_wave, wave_identifiers = group_features_by_waves(X.columns)
    X_reshaped = reshape_data(X_scaled, features_by_wave, wave_identifiers, X.columns)

    best_params, best_score, all_results = grid_search(X_reshaped, y, HYPERPARAMETER_GRID, KFOLDS)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best F1 Score: {best_score:.4f}")

    save_results_to_file(all_results, "results.txt")

if __name__ == "__main__":
    main()
