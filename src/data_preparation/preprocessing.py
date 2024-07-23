import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LongitudinalDataset:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_groups = None

    def load_data_target_train_test_split(self, target_column: str, test_size: float = 0.2, random_state: int = None) -> None:
        self.data = pd.read_csv(self.file_path)
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def setup_features_group(self, group_name: str) -> None:
        # This is a placeholder function. Implement your feature grouping logic here.
        # For now, it assumes each column is a separate group.
        self.feature_groups = [[i] for i in range(self.X_train.shape[1])]
        
    def get_feature_groups(self):
        return self.feature_groups