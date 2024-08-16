# Longitudinal MLP

This script implements a Multi-Layer Perceptron (MLP) tailored for longitudinal data analysis. It is designed to handle temporal data with grouped features, allowing for separate processing of feature groups before combining them for binary classification tasks.

## Overview

The `Longitudinal MLP` script performs the following key tasks:

1. **Data Loading and Preprocessing:**
   - Loads longitudinal data from a specified path.
   - Handles missing values, ensuring all entries are numeric.
   - Splits the data into training and test sets.
   - Normalizes the data using Min-Max scaling.
   - Configures the input size based on the dataset.

2. **Model Architecture:**
   - A custom MLP model is built, with separate hidden layers for each feature group.
   - Outputs from the hidden layers are concatenated and passed through a dropout layer before reaching the output layer.
   - The model uses binary cross-entropy as the loss function, suitable for binary classification tasks.

3. **Model Training and Evaluation:**
   - The model is trained using a grid search for hyperparameter optimization.
   - The script evaluates the model's performance using various metrics, including accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC score.

## Installation and Setup

### Prerequisites

**Python 3.9.x is required to run the script.**

Ensure you have the required Python packages installed. You can install them using `pip`:

```bash
pip install numpy pandas torch scikit-learn skorch scikit-longitudinal python-dotenv
```
- Be sure to install Scikit-Longitudinal correctly by following the instructions found [here](https://simonprovost.github.io/scikit-longitudinal/quick-start/#installation).

### Environment Variables

The script uses environment variables to load the dataset path and target wave. Create a `.env` file in the root directory of your project with the following content:

```env
DATASET_PATH=<path_to_your_dataset>
TARGET_WAVE=<target_wave_column_name>
```

### Running the Script

Execute the script to train and evaluate the model:

```bash
python long_mlp_gridsearch.py
```

## Key Components

### Data Preprocessing

- **Missing Value Handling:** Replaces missing values (`?`) with `NaN`, coerces data to numeric types, and fills remaining `NaN` values with `0`.
- **Normalization:** Features are normalized using Min-Max scaling, ensuring that they fall within a specific range, which aids in model convergence.

### Model Architecture

- **Feature Grouping:** Features are grouped and processed independently through separate hidden layers.
- **Dropout Regularization:** Dropout layers are applied after the hidden layers to prevent overfitting.
- **Output Layer:** The final output layer generates a single output neuron representing the probability of the positive class.

### Training and Evaluation

- **Grid Search:** The script uses `GridSearchCV` to optimize hyperparameters, including learning rate, number of epochs, hidden layer size, and dropout rate.
- **Performance Metrics:** The model is evaluated using accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC score.

## Example Output

The script logs key metrics during training and evaluation, such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC-AUC Score**

Example log output:
```plaintext
INFO: Gridsearching...
INFO: Best score: 0.850, Best params: {'lr': 0.3, 'max_epochs': 200, 'module__hidden_size': 128, 'module__dropout_rate': 0.4}
INFO: Starting model training...
INFO: Evaluating model...
INFO: Accuracy: 0.7845
INFO: Precision: 0.8123
INFO: Recall: 0.7634
INFO: F1 Score: 0.7872
INFO: Confusion Matrix:
[[1135  180]
 [ 240  655]]
INFO: ROC-AUC Score: 0.8504
INFO: Model target: <target_wave_column_name>
```

## Customization

- **Hyperparameters:** Modify hyperparameters such as `HIDDEN_SIZE`, `DROPOUT_RATE`, `MAX_EPOCHS`, and `LR` directly in the script.
- **Feature Grouping:** Customize the feature grouping based on your dataset using the `setup_features_group` method in the `LongitudinalDataset` class.