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
DATASET_PATH = "../../data/angina_dataset.csv"
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
    print(f"Wave {wave} features: {wave_features}")
print(f"X reshaped shape: {X_reshaped.shape}")
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
# Define the CNN model
class CNNModel(Module):
    def __init__(self, input_dim, num_channels, kernel_size, pool_size, dropout=0.0):
        super(CNNModel, self).__init__()
        self.conv1 = Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=kernel_size, padding=1)
        self.pool = MaxPool1d(pool_size)
        self.dropout = Dropout(dropout)
        
        # Calculate the output size after Conv1d and MaxPool1d
        conv_output_size = ((n_timesteps - kernel_size + 2 * 1) // 1 + 1) // pool_size
        self.feature_dim = num_channels * conv_output_size

        self.fc = Linear(self.feature_dim, DENSE_UNITS)
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
from itertools import product

# Define hyperparameter grid
learning_rates = [0.001, 0.01]  # Include a broader range of learning rates
dropout_rates = [0.2, 0.5]  # Include an additional dropout rate for more variety
num_channels_options = [64, 128]  # Different numbers of output channels for the Conv1d layer
kernel_size_options = [3, 5]  # Different kernel sizes for the Conv1d layer

# All combinations of hyperparameters
hyperparameter_grid = list(product(learning_rates, dropout_rates, num_channels_options, kernel_size_options))
from sklearn.model_selection import KFold

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
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
    return f1, conf_matrix, recall, precision

def grid_search(train_data, val_data, hyperparameter_grid, device, n_epochs=10):
    best_hyperparameters = None
    best_f1 = float('-inf')
    best_conf_matrix = None
    best_precision = None
    best_recall = None

    for lr, dropout, num_channels, kernel_size in hyperparameter_grid:
        print(f"Training with learning rate: {lr}, dropout: {dropout}, channels: {num_channels}, kernel size: {kernel_size}")
        model = CNNModel(input_dim, num_channels, kernel_size, POOL_SIZE, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()
        
        # Prepare data loaders
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train and evaluate
        f1, conf_matrix, recall, precision = train_validate_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
        
        # Save the best hyperparameters based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_conf_matrix = conf_matrix
            best_hyperparameters = (lr, dropout, num_channels, kernel_size)
            best_precision = precision
            best_recall = recall
            print(f"Found new best hyperparameters: {best_hyperparameters}")

    return best_hyperparameters, best_f1, best_conf_matrix, best_precision, best_recall


# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_fold_results = []

best_results_per_fold = []

for fold, (train_index, test_index) in enumerate(kf.split(X_reshaped)):
    print(f'Fold {fold + 1}')
    
    # Split data into train and test sets for this fold
    X_train_fold, X_test_fold = X_reshaped[train_index], X_reshaped[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    # Further split train into learning and validation sets
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_fold, y_train_fold, test_size=VALIDATION_SPLIT, random_state=42)
    
    # Convert to TensorDatasets
    train_data_sub = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(y_train_sub.values, dtype=torch.float32))
    val_data_sub = TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32), torch.tensor(y_val_sub.values, dtype=torch.float32))
    
    # Perform grid search on the learning and validation set
    best_hyperparameters, best_f1, best_conf_matrix, best_preicison, best_recall = grid_search(train_data_sub, val_data_sub, hyperparameter_grid, device, n_epochs=MAX_EPOCHS)
    
    print(f"Best Hyperparameters for Fold {fold + 1}: {best_hyperparameters}")
    
    best_results_per_fold.append([best_hyperparameters, best_f1, best_conf_matrix, best_preicison, best_recall])
    
    # Unpacking the correct hyperparameters for CNNModel
    lr, dropout, num_channels, kernel_size = best_hyperparameters
    print(f"Training on full training set with best hyperparameters: {best_hyperparameters}")
    model = CNNModel(input_dim, num_channels, kernel_size, POOL_SIZE, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Prepare final train and test loaders
    train_data_final = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold.values, dtype=torch.float32))
    test_data_final = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(y_test_fold.values, dtype=torch.float32))
    
    train_loader_final = DataLoader(train_data_final, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_final = DataLoader(test_data_final, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train on the full training set
    train_model(model, train_loader_final, criterion, optimizer, MAX_EPOCHS, device)
    
    # Evaluate on the test set
    results = evaluate_model(model, test_loader_final, device)
    

list_of_dicts = []
def format_confusion_matrix(conf_matrix):
    return f"[[{conf_matrix[0][0]}, {conf_matrix[0][1]}]\n [{conf_matrix[1][0]}, {conf_matrix[1][1]}]]"

for fold, i in enumerate(best_results_per_fold):
    cleaned_hyperparameters = i[0]
    cleaned_conf_matrix = i[2]
    # i[1] round to 4 decimal places
    converted_f1 = round(i[1], 4)
    convert_precision = round(i[3], 4)
    convert_recall = round(i[4], 4)
    print(cleaned_hyperparameters)
    #print(format_confusion_matrix(cleaned_conf_matrix))
    list_of_dicts.append({'Hyperparameters': cleaned_conf_matrix})

# Print the DataFrame without the index
print(pd.DataFrame(list_of_dicts).to_string(index=False))

print(f"Dataset name is: {DATASET_PATH}")