import pandas as pd
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import LexicoGradientBoostingClassifier

print("starting...")

# Sample dataset path
dataset_path = '/data/dataset_target.csv'

# Load the dataset using pandas to clean it
data = pd.read_csv(dataset_path)

# Replace '?' with 0
data.replace('?', 0, inplace=True)

# Convert all columns to numeric, coercing errors to NaN (although there should be no '?')
data = data.apply(pd.to_numeric, errors='coerce')

# Ensure no NaN values are left (if any)
data.fillna(0, inplace=True)

# Save the cleaned data to a temporary CSV file
cleaned_dataset_path = './cleaned_dataset.csv'
data.to_csv(cleaned_dataset_path, index=False)

# Load and prepare the dataset
dataset = LongitudinalDataset(cleaned_dataset_path)
dataset.load_data_target_train_test_split(
    target_column="class_target_w4",
)

print("Dataset loaded...")

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="Elsa")

print("Features group set...")

# Initialize and train the model
model = LexicoGradientBoostingClassifier(
    features_group=dataset.feature_groups(),
    threshold_gain=0.00015
)

print("Model initialized...")

model.fit(dataset.X_train, dataset.y_train)

print("Model fitted...")

# Make predictions
y_pred = model.predict(dataset.X_test)
print("Predictions: ", y_pred)
print("Test: ", dataset.y_test)
