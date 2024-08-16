# GRU-Based Longitudinal Data Model

This script implements a Gated Recurrent Unit (GRU) model designed for handling longitudinal data, particularly for binary classification tasks. It processes a dataset, prepares it for recurrent neural network training, and performs a hyperparameter search with cross-validation.

## Overview

The script performs the following tasks:

1. **Data Loading and Preprocessing:**
   - Loads the dataset from a specified path.
   - Preprocesses the data by handling missing values and converting features to numeric types.
   - Splits the dataset into training and test sets and normalizes the data.
   - Organizes features by wave (time period) for longitudinal processing.

2. **Model Definition:**
   - Defines a GRU-based model with optional bidirectional layers and dropout for regularization.
   - The model is designed to handle sequences of data with multiple timesteps (waves).

3. **Training and Evaluation:**
   - The model is trained using the processed data, with performance evaluated on various metrics including precision, recall, F1 score, ROC-AUC, and AUPRC.
   - A grid search is performed to identify the best hyperparameters, utilizing K-Fold cross-validation.

## Setup and Installation

1. **Install Dependencies:**
   Ensure you have the required Python packages installed. You can install them using `pip`:

   ```bash
   pip install numpy pandas torch scikit-learn
   ```

2. **Prepare Your Dataset:**
   Place your dataset in the specified path and ensure it follows the expected format. Update the `DATASET_PATH` constant in the script with the correct file path.

3. **Run the Script:**
   Execute the script to train the GRU model and evaluate its performance:

   ```bash
   python longitudinal_gru.py
   ```

## Key Components

### Data Preprocessing

- **Missing Value Handling:** Missing values are replaced with `NaN`, and the data is coerced to numeric types. Remaining `NaN` values are filled with `0`.
- **Feature Grouping:** Features are grouped by waves (timesteps), allowing the model to process sequential data effectively.

### Model Architecture

- **GRU Layers:** The model consists of GRU layers with optional bidirectional configuration and dropout.
- **Output Layer:** A dense output layer with a sigmoid activation function provides the binary classification output.

### Training and Evaluation

- **Hyperparameter Search:** A grid search is performed to find the best combination of learning rate, dropout rate, number of GRU layers, and GRU units.
- **K-Fold Cross Validation:** The script uses K-Fold cross-validation to evaluate model performance across different data splits.

### Metrics

- **Precision, Recall, F1 Score:** Standard classification metrics for evaluating binary classification performance.
- **ROC-AUC and AUPRC:** Metrics that consider the trade-off between true positive rate and false positive rate, as well as precision-recall trade-offs.

## Example Output

During execution, the script logs key metrics and hyperparameter configurations, such as:

```plaintext
Fold 1
Training with learning rate: 0.001, dropout: 0.3, layers: 2, units: 100
Found new best hyperparameters: (0.001, 0.3, 2, 100)
Epoch 1/100, Loss: 0.6931
...
Test Precision: 0.8123
Test Recall: 0.7634
Test F1 Score: 0.7872
Test ROC-AUC Score: 0.8504
Test AUPRC Score: 0.8207
Confusion Matrix:
[[1135  180]
 [ 240  655]]
Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.86      0.85      1315
           1       0.79      0.77      0.78       895
    accuracy                           0.83      2210
   macro avg       0.82      0.81      0.82      2210
weighted avg       0.83      0.83      0.83      2210
```

## Customization

- **Hyperparameters:** Modify the hyperparameters such as `LEARNING_RATE`, `DROPOUT`, `GRU_UNITS`, and `NUM_LAYERS` directly in the script.
- **Wave and Feature Handling:** Adjust how features are grouped and processed by waves as needed for your specific dataset.

## Notes

- The script is designed to handle binary classification tasks on longitudinal data with multiple time points (waves).
- Ensure that the dataset follows the expected format, particularly with respect to the naming of wave-specific features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```

This `README.md` file provides a clear description of the GRU-based longitudinal data model script, explaining its purpose, setup, and usage. It covers data preprocessing, model architecture, training and evaluation, and customization options, making it easy for users to understand and apply the script to their own longitudinal datasets.