# Longitudinal MLP

This script implements a Multi-Layer Perceptron (MLP) designed specifically for handling longitudinal datasets, where temporal vectors need to be processed effectively. The model architecture is designed to accommodate feature groups, allowing for the separation and independent processing of different sets of features, which are then combined to predict binary outcomes.

## Overview

1. **Data Loading and Preprocessing:** 
   - Loads a longitudinal dataset from a specified path.
   - Handles missing values and ensures that all features are numeric.
   - Splits the dataset into training and test sets.
   - Normalizes the data using Min-Max scaling.
   - Configures the input size based on the dataset.

2. **Model Architecture:**
   - A custom MLP model is constructed, where each feature group is processed by a separate hidden layer.
   - The outputs of these hidden layers are concatenated and passed through a dropout layer for regularization before reaching the final output layer.
   - The model is trained using binary cross-entropy loss.

3. **Model Training and Evaluation:**
   - The model is trained using the training data.
   - Metrics are calculated on the test data to evaluate the model's performance, including accuracy, precision, recall, F1 score, confusion matrix, ROC-AUC score, and AUPRC.

## Installation and Setup

1. **Install Dependencies:**
   Ensure you have the required Python packages installed. You can install them using `pip`:

   ```bash
   pip install numpy pandas torch scikit-learn skorch scikit-longitudinal python-dotenv
   ```

2. **Environment Variables:**
   The script uses environment variables to load the dataset path and target wave. Create a `.env` file in the root directory of your project with the following content:

   ```env
   DATASET_PATH=<path_to_your_dataset>
   TARGET_WAVE=<target_wave_column_name>
   ```

3. **Run the Script:**
   Execute the script to train and evaluate the model:

   ```bash
   python longitudinal_mlp.py
   ```

## Key Components

### Data Preprocessing

- **Missing Value Handling:** Missing values are replaced with `NaN`, and all data is coerced to numeric types. Any remaining `NaN` values are filled with `0`.
- **Normalization:** Features are normalized using Min-Max scaling to ensure that they fall within a specific range, which helps in model convergence.

### Model Architecture

- **Feature Grouping:** Features are grouped and processed independently through separate hidden layers.
- **Dropout Regularization:** A dropout layer is applied after the hidden layers to prevent overfitting.
- **Output Layer:** The final output layer produces a single output neuron that represents the probability of the positive class.

### Training and Evaluation

- **Grid Search:** The script includes a basic grid search functionality for hyperparameter optimization, focusing on learning rate, number of epochs, hidden layer size, and dropout rate.
- **Performance Metrics:** After training, the model's performance is evaluated using a comprehensive set of metrics, including precision, recall, F1 score, ROC-AUC, and AUPRC.

## Example Output

The script logs key metrics during training and evaluation, such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC-AUC Score**
- **AUPRC Score**
- **Classification Report**

Example log output:
```plaintext
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
INFO: AUPRC Score: 0.8207
INFO: Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.86      0.85      1315
           1       0.79      0.77      0.78       895
    accuracy                           0.83      2210
   macro avg       0.82      0.81      0.82      2210
weighted avg       0.83      0.83      0.83      2210
```

## Customization

- **Hyperparameters:** You can modify the hyperparameters such as `HIDDEN_SIZE`, `DROPOUT_RATE`, `MAX_EPOCHS`, and `LR` directly in the script.
- **Feature Grouping:** The feature grouping can be customized based on your dataset by adjusting the `setup_features_group` method in the `LongitudinalDataset` class.

## Notes

- The script is designed to handle binary classification tasks on longitudinal data.
- Ensure that the dataset follows the expected format, especially regarding the feature grouping and target wave column.
- The script uses a fixed random seed for reproducibility of results.
