from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from scikit_longitudinal.data_preparation import LongitudinalDataset
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH: str = "./data/hbp_dataset.csv"
TARGET_WAVE: str = "class_hbp_w8"
# Hyperparameters
INPUT_SIZE: int = None  # Will be set after loading data 
HIDDEN_SIZE: int = 128  # Number of neurons in each hidden layer
MAX_EPOCHS: int = 100  # Increased maximum number of epochs for training
LR: float = 0.001  # Adjusted learning rate for optimization
DROPOUT_RATE: float = 0.4  # Dropout rate for regularization
KFOLDS: int = 10  # Number of folds for cross-validation

# Set seeds for reproducibility
np.random.seed(42)

def preprocess_data(X: np.ndarray) -> np.ndarray:
    X_df: pd.DataFrame = pd.DataFrame(X)
    X_df.replace('?', np.nan, inplace=True)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.fillna(0, inplace=True)
    return X_df.values

def load_and_preprocess_data() -> tuple:
    dataset: LongitudinalDataset = LongitudinalDataset(DATASET_PATH)
    dataset.load_data_target_train_test_split(target_column=TARGET_WAVE, random_state=42, test_size=0.1, remove_target_waves=True)
    dataset.setup_features_group("elsa")

    X_train: np.ndarray = preprocess_data(dataset.X_train)
    X_test: np.ndarray = preprocess_data(dataset.X_test)
    y_train: np.ndarray = pd.to_numeric(dataset.y_train, errors='coerce').values
    y_test: np.ndarray = pd.to_numeric(dataset.y_test, errors='coerce').values

    # Normalize data
    scaler: MinMaxScaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    global INPUT_SIZE
    INPUT_SIZE = X_train.shape[1]

    return X_train, X_test, y_train, y_test, dataset.feature_groups()

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray) -> None:
    logger.info("Starting cross-validation...")
    
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    
    fold_results = []

    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (128,), (256,)],
        'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
        'alpha': [0.001, 0.01, 0.2, 0.4]
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        logger.info(f"Starting fold {fold}")
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Define the MLP Classifier with GridSearchCV
        mlp = MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=MAX_EPOCHS,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10
        )

        search = RandomizedSearchCV(estimator=mlp, param_grid=param_grid, scoring='f1', cv=KFOLDS, n_jobs=-1, verbose=1)
        search.fit(X_train_fold, y_train_fold)

        best_mlp = search.best_estimator_
        logger.info(f"Best parameters for fold {fold}: {search.best_params_}")

        # Evaluate the model on the test set of this fold
        y_predictions = best_mlp.predict(X_test_fold)
        
        # Calculate additional metrics
        precision = precision_score(y_test_fold, y_predictions)
        recall = recall_score(y_test_fold, y_predictions)
        f1 = f1_score(y_test_fold, y_predictions)
        conf_matrix = confusion_matrix(y_test_fold, y_predictions)
        roc_auc = roc_auc_score(y_test_fold, y_predictions)

        # Calculate precision-recall curve and AUPRC
        y_prob = best_mlp.predict_proba(X_test_fold)[:, 1]
        precision_recall = precision_recall_curve(y_test_fold, y_prob)
        auprc = auc(precision_recall[1], precision_recall[0])
        report = classification_report(y_test_fold, y_predictions)

        # Log fold results
        logger.info(f"Results for fold {fold}:")
        logger.info(f'Best F1 Score: {f1:.4f}')
        logger.info(f'Precision: {precision:.4f}')
        logger.info(f'Recall: {recall:.4f}')
        logger.info(f'Confusion Matrix:\n{conf_matrix}')
        logger.info(f'ROC-AUC Score: {roc_auc:.4f}')
        logger.info(f"AUPRC Score: {auprc:.4f}")
        logger.info(f"Classification Report: {report}")

        fold_results.append({
            'fold': fold,
            'best_params': search.best_params_,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'auprc': auprc
        })

    # Summary of results
    logger.info("Cross-validation completed.")
    for result in fold_results:
        print(f"Fold {result['fold']}\nF1: {result['f1_score']:.4f}\nPrecision: {result['precision']:.4f}\nRecall: {result['recall']:.4f}\nBest Params: {result['best_params']}\nConfusion Matrix: {result['confusion_matrix']}")
        # create a file using the path name and write the results to it
        with open(f"trad-mlp-gs-{TARGET_WAVE}.txt", "a") as file:
            file.write(f"Fold {result['fold']}\nF1: {result['f1_score']:.4f}\nPrecision: {result['precision']:.4f}\nRecall: {result['recall']:.4f}\nBest Params: {result['best_params']}\nConfusion Matrix: {result['confusion_matrix']}\n")

def main() -> None:
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    train_and_evaluate_model(X_combined, y_combined)

if __name__ == "__main__": 
    main()
