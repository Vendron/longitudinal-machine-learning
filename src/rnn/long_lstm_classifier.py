import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from torch.nn import Module, LSTM, Sequential, Linear, Sigmoid, BCEWithLogitsLoss
from torch.nn.functional import dropout
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, classification_report, precision_recall_curve
)
from scikit_longitudinal.data_preparation import LongitudinalDataset

# Constants
DATASET_PATH = 'DATASET.csv'
TARGET_NAME = "heartattack"  # Replace with the correct target name from your dataset
TARGET_WAVE = f"class_{TARGET_NAME}_w8"

def preprocess_data(data):
    """Preprocess the data."""
    data = pd.to_numeric(data, errors='coerce')
    data.fillna(data.mean(), inplace=True)
    return data.values

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, dataset.feature_groups()

def reshape_data_for_lstm(X, feature_groups):
    """Reshape the data for LSTM."""
    n_samples = X.shape[0]
    n_timesteps = len(feature_groups)
    max_features = max(len(features) for features in feature_groups.values())

    X_reshaped = np.zeros((n_samples, n_timesteps, max_features))
    for i, (wave, features) in enumerate(feature_groups.items()):
        indices = [X.columns.get_loc(f) for f in features]
        X_reshaped[:, i, :len(indices)] = X[:, indices]

    return X_reshaped

def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    y_scores = model.predict(X_test)
    y_pred = (y_scores > 0.5).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auprc = auc(recall, precision)
    report = classification_report(y_test, y_pred)

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC-AUC Score: {roc_auc:.4f}')
    print(f'AUPRC Score: {auprc:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{report}')

def main():
    """Main function to run the workflow."""
    X_train, X_test, y_train, y_test, feature_groups = load_and_prepare_data(DATASET_PATH, TARGET_WAVE)
    
    X_train_reshaped = reshape_data_for_lstm(X_train, feature_groups)
    X_test_reshaped = reshape_data_for_lstm(X_test, feature_groups)
    
    model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
    
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2)
    
    evaluate_model(model, X_test_reshaped, y_test)

if __name__ == "__main__":
    main()
