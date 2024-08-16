import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from scikit_longitudinal.data_preparation import LongitudinalDataset
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Linear, Sigmoid, Sequential, BCEWithLogitsLoss, ModuleList
from torch import Tensor, cat
from torch.nn.functional import dropout
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import logging 
from functools import partial

# Setup logging    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH: str = "./data/parkinsons_dataset.csv"
TARGET_WAVE: str = "class_parkinsons_w8"
# Hyperparameters
INPUT_SIZE: int = None  # Will be set after loading data 
HIDDEN_SIZE: int = 128  # Number of neurons in each hidden layer
OUTPUT_SIZE: int = 1  # Number of neurons in the output layer
DROPOUT_RATE: float = 0.4  # Dropout rate for regularization
MAX_EPOCHS: int = 200  # Maximum number of epochs for training
LR: float = 0.3  # Learning rate for optimization
ITERATOR_TRAIN_SHUFFLE: bool = True  # Shuffle the training data before each epoch
TRAIN_SPLIT: None = None  # Use the entire training data for training
VERBOSE: int = 1  # Verbosity level for logging during training
CRITERION: Module = BCEWithLogitsLoss()  # Binary Cross-Entropy loss function
KFOLDS: int = 10  # Number of folds for cross-validation

# Set seeds for reproducibility
torch.manual_seed(42)
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

class LongitudinalMLPModule(Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_rate: float, features_group: List[List[int]]) -> None:
        super(LongitudinalMLPModule, self).__init__()
        self.features_group: List[List[int]] = features_group
        self.hidden_layers: ModuleList = ModuleList()
        self.dropout_rate: float = dropout_rate

        for group in features_group:
            group_size: int = len(group)
            self.hidden_layers.append(Sequential(
                Linear(group_size, hidden_size),
                Sigmoid()
            ))

        self.output_layer: Linear = Linear(hidden_size * len(features_group), output_size) 

    def forward(self, X: Tensor) -> Tensor:
        group_outputs: List[Tensor] = []

        for idx, group in enumerate(self.features_group):
            group_x: Tensor = X[:, group]
            group_output: Tensor = self.hidden_layers[idx](group_x)
            group_outputs.append(group_output)

        concatenated_outputs: Tensor = cat(group_outputs, dim=1)
        concatenated_outputs = dropout(concatenated_outputs, p=self.dropout_rate, training=self.training)
        output: Tensor = self.output_layer(concatenated_outputs)

        return output

def create_search_space() -> dict:
    lr_range: List[float] = [0.001, 0.01, 0.3]
    max_epochs_range: List[int] = [100]
    hidden_size_range: List[int] = [32, 128]
    dropout_rate_range: List[float] = [0.3, 0.4]

    search_space: dict = {
        'lr': lr_range,
        'max_epochs': max_epochs_range,
        'module__hidden_size': hidden_size_range,
        'module__dropout_rate': dropout_rate_range,
    }

    return search_space

def save_checkpoint(output_file: str, all_fold_results: list, current_fold: int, best_f1_results: Dict[str, Any], best_auc_results: Dict[str, Any]):
    checkpoint_data = {
        'all_fold_results': all_fold_results,
        'current_fold': current_fold,
        'best_f1_results': best_f1_results,
        'best_auc_results': best_auc_results
    }
    with open(output_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(output_file: str):
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            return (checkpoint_data['all_fold_results'], 
                    checkpoint_data['current_fold'],
                    checkpoint_data['best_f1_results'],
                    checkpoint_data['best_auc_results'])
    return [], 0, {}, {}

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, features_group: List[List[int]], output_file: str, checkpoint_file: str) -> None:
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_fold_results, start_fold, best_f1_results, best_auc_results = load_checkpoint(checkpoint_file)
    
    with open(output_file, 'a') as f:
        for fold, (train_index, test_index) in enumerate(kf.split(X), start=start_fold):
            f.write(f'\nFold {fold + 1}\n')
            print(f'Fold {fold + 1}')
            
            # Split data into training and test sets for this fold
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            # Further split train into learning and validation sets
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_fold, y_train_fold, test_size=0.1, random_state=42)
            
            model_partial: Module = partial(LongitudinalMLPModule, output_size=OUTPUT_SIZE, features_group=features_group)
            
            search_space: Dict = create_search_space()
            
            auc_callback: EpochScoring = EpochScoring(scoring='roc_auc', lower_is_better=False)
            
            model: NeuralNetBinaryClassifier = NeuralNetBinaryClassifier(
                module=model_partial,
                max_epochs=MAX_EPOCHS,
                lr=LR,
                iterator_train__shuffle=ITERATOR_TRAIN_SHUFFLE,
                train_split=TRAIN_SPLIT,
                verbose=0,  # Deactivate verbose logging for GridSearchCV
                criterion=CRITERION,
                callbacks=[auc_callback]
            )
            
            # Perform grid search on the learning and validation set
            search: GridSearchCV = GridSearchCV(estimator=model, param_grid=search_space, cv=10, refit=True, scoring='f1', n_jobs=-1, verbose=2)
            logger.info(f"Gridsearching for Fold {fold + 1}...")
            search.fit(X_train_sub.astype(np.float32), y_train_sub.astype(np.float32))
            logger.info(f"Best score: {search.best_score_:.3f}, Best params: {search.best_params_}")
            f.write(f"Best score: {search.best_score_:.3f}, Best params: {search.best_params_}\n")
            
            # Get the best model
            best_model: NeuralNetBinaryClassifier = search.best_estimator_
            
            # Train on the full training set of the fold
            logger.info(f"Training best model for Fold {fold + 1}...")
            best_model.fit(X_train_fold.astype(np.float32), y_train_fold.astype(np.float32))
            
            # Evaluate on the test set of the fold
            logger.info(f"Evaluating model for Fold {fold + 1}...")
            y_predictions: np.ndarray  = best_model.predict(X_test_fold.astype(np.float32))
            accuracy: float = np.mean(y_predictions == y_test_fold)
            logger.info(f'Accuracy for Fold {fold + 1}: {accuracy:.4f}')
            
            # Calculate additional metrics
            precision: float = precision_score(y_test_fold, y_predictions)
            recall: float = recall_score(y_test_fold, y_predictions)
            f1: float = f1_score(y_test_fold, y_predictions)
            conf_matrix: np.ndarray = confusion_matrix(y_test_fold, y_predictions)
            roc_auc: float = roc_auc_score(y_test_fold, y_predictions)
            
            logger.info(f'Precision for Fold {fold + 1}: {precision:.4f}')
            logger.info(f'Recall for Fold {fold + 1}: {recall:.4f}')
            logger.info(f'F1 Score for Fold {fold + 1}: {f1:.4f}')
            logger.info(f'Confusion Matrix for Fold {fold + 1}:\n{conf_matrix}')
            logger.info(f'ROC-AUC Score for Fold {fold + 1}: {roc_auc:.4f}')
            
            f.write(f'Precision for Fold {fold + 1}: {precision:.4f}\n')
            f.write(f'Recall for Fold {fold + 1}: {recall:.4f}\n')
            f.write(f'F1 Score for Fold {fold + 1}: {f1:.4f}\n')
            f.write(f'Confusion Matrix for Fold {fold + 1}:\n{conf_matrix}\n')
            f.write(f'ROC-AUC Score for Fold {fold + 1}: {roc_auc:.4f}\n')
            
            # Save results
            all_fold_results.append({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            })
            
            # Check for the best F1 and AUC scores across all folds and update
            if not best_f1_results or f1 > best_f1_results['f1']:
                best_f1_results = {
                    'fold': fold + 1,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'hyperparams': search.best_params_
                }
                
            if not best_auc_results or roc_auc > best_auc_results['roc_auc']:
                best_auc_results = {
                    'fold': fold + 1,
                    'roc_auc': roc_auc,
                    'hyperparams': search.best_params_
                }
            
            # Save checkpoint after each fold
            save_checkpoint(checkpoint_file, all_fold_results, fold + 1, best_f1_results, best_auc_results)
        
        # Write the best results
        logger.info(f'BEST F1 SCORE: {best_f1_results["f1"]:.4f}, '
                    f'RECALL: {best_f1_results["recall"]:.4f}, '
                    f'PRECISION: {best_f1_results["precision"]:.4f}, '
                    f'HYPERPARAMS: {best_f1_results["hyperparams"]}, '
                    f'FOLD: {best_f1_results["fold"]}')
        
        f.write(f'BEST F1 SCORE: {best_f1_results["f1"]:.4f}, '
                f'RECALL: {best_f1_results["recall"]:.4f}, '
                f'PRECISION: {best_f1_results["precision"]:.4f}, '
                f'HYPERPARAMS: {best_f1_results["hyperparams"]}, '
                f'FOLD: {best_f1_results["fold"]}\n')
        
        logger.info(f'BEST ROC-AUC SCORE: {best_auc_results["roc_auc"]:.4f}, '
                    f'HYPERPARAMS: {best_auc_results["hyperparams"]}, '
                    f'FOLD: {best_auc_results["fold"]}')
        
        f.write(f'BEST ROC-AUC SCORE: {best_auc_results["roc_auc"]:.4f}, '
                f'HYPERPARAMS: {best_auc_results["hyperparams"]}, '
                f'FOLD: {best_auc_results["fold"]}\n')
        
        logger.info('All results:')
        logger.info(all_fold_results)   
        logger.info(f'BEST PARAMS: {search.best_params_}')
        f.write(f'All results: {all_fold_results}\n')
        f.write(f'BEST PARAMS: {search.best_params_}\n')

def main() -> None:
    dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    output_file = f'{dataset_name}_mlp_results.txt'
    checkpoint_file = f'{dataset_name}_mlp_checkpoint.pkl'

    X_train, X_test, y_train, y_test, features_group = load_and_preprocess_data()
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    train_and_evaluate_model(X_combined, y_combined, features_group, output_file, checkpoint_file)

if __name__ == "__main__":
    main()
