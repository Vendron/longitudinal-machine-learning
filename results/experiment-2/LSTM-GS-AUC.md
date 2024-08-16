# RNN LSTM Grid Search KFold AUC Results

This section summarises the AUC scores obtained from a grid search with K-Fold cross-validation using RNN LSTM models. The experiments were conducted with varying learning rates, dropout rates, the number of LSTM layers, and LSTM units. The results provide insights into how these hyperparameters influence model performance across different datasets.

**Hyperparameters Grid:**
- **Learning Rates:** [0.001, 0.01]
- **Dropout Rates:** [0.2, 0.5]
- **Number of Layers:** [1, 2]
- **LSTM Units:** [50, 100]

### Angina

The model generally performed well in terms of AUC scores, though the confusion matrices indicate a complete lack of positive class detection, which points to a severe class imbalance issue.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.7521    | [[616, 0], [23, 0]]   | (0.01, 0.5, 1, 100)                |
| 2    | 0.7303    | [[618, 0], [21, 0]]   | (0.01, 0.2, 2, 50)                 |
| 3    | 0.7511    | [[614, 0], [25, 0]]   | (0.01, 0.5, 2, 50)                 |
| 4    | 0.7435    | [[621, 0], [18, 0]]   | (0.001, 0.2, 2, 50)                |
| 5    | 0.7342    | [[616, 0], [23, 0]]   | (0.001, 0.2, 1, 50)                |
| 6    | 0.6918    | [[614, 0], [25, 0]]   | (0.01, 0.2, 1, 100)                |
| 7    | 0.6995    | [[618, 0], [21, 0]]   | (0.01, 0.2, 1, 100)                |
| 8    | 0.7247    | [[625, 0], [14, 0]]   | (0.001, 0.5, 1, 100)               |
| 9    | 0.7324    | [[620, 0], [19, 0]]   | (0.001, 0.5, 1, 100)               |
| 10   | 0.6705    | [[613, 0], [26, 0]]   | (0.01, 0.2, 2, 50)                 |

### Arthritis

The Arthritis dataset showed more variability in AUC scores across different folds. The confusion matrices suggest some detection of the positive class, but with significant false negatives and positives.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.6758    | [[316, 58], [174, 91]] | (0.001, 0.5, 2, 50)               |
| 2    | 0.6706    | [[334, 41], [186, 78]] | (0.01, 0.5, 2, 100)               |
| 3    | 0.6952    | [[315, 62], [158, 104]]| (0.001, 0.2, 1, 50)               |
| 4    | 0.6543    | [[321, 50], [180, 88]] | (0.001, 0.5, 2, 100)              |
| 5    | 0.6758    | [[308, 52], [191, 88]] | (0.001, 0.5, 2, 50)               |
| 6    | 0.6584    | [[309, 48], [183, 99]] | (0.001, 0.5, 2, 50)               |
| 7    | 0.6668    | [[330, 43], [182, 84]] | (0.001, 0.2, 2, 50)               |
| 8    | 0.6565    | [[287, 71], [185, 96]] | (0.001, 0.5, 1, 50)               |
| 9    | 0.6801    | [[347, 46], [183, 63]] | (0.01, 0.5, 2, 50)                |
| 10   | 0.6694    | [[322, 44], [187, 86]] | (0.01, 0.5, 2, 100)               |

### Cataract

The results for the Cataract dataset show moderate AUC scores, but the confusion matrices indicate a relatively high number of false negatives.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.7398    | [[382, 37], [165, 55]] | (0.001, 0.5, 1, 100)               |
| 2    | 0.6998    | [[386, 52], [148, 53]] | (0.001, 0.2, 1, 50)                |
| 3    | 0.7192    | [[373, 39], [166, 61]] | (0.001, 0.2, 1, 50)                |
| 4    | 0.6945    | [[376, 61], [149, 53]] | (0.01, 0.2, 1, 50)                 |
| 5    | 0.7043    | [[399, 34], [166, 40]] | (0.001, 0.5, 1, 100)               |
| 6    | 0.6838    | [[363, 48], [170, 58]] | (0.001, 0.2, 1, 50)                |
| 7    | 0.6994    | [[386, 49], [155, 49]] | (0.001, 0.5, 1, 100)               |
| 8    | 0.6850    | [[359, 60], [166, 54]] | (0.001, 0.5, 2, 100)               |
| 9    | 0.6892    | [[379, 55], [143, 62]] | (0.001, 0.2, 1, 100)               |
| 10   | 0.7170    | [[372, 38], [173, 56]] | (0.001, 0.5, 1, 50)                |

### Diabetes

The Diabetes dataset had relatively high AUC scores, indicating the model's effectiveness in distinguishing between classes, though some folds showed issues with false positives and negatives.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.8239    | [[602, 19], [59, 30]] | (0.01, 0.5, 1, 50)                 |
| 2    | 0.8066    | [[592, 18], [65, 35]] | (0.001, 0.2, 1, 100)               |
| 3    | 0.8363    | [[582, 22], [68, 38]] | (0.01, 0.2, 1, 50)                 |
| 4    | 0.4973    | [[613, 0], [97, 0]]   | (0.01, 0.2, 2, 50)                 |
| 5    | 0.7922    | [[605, 16], [70, 19]] | (0.01, 0.2, 1, 50)                 |
| 6    | 0.7952    | [[603, 11], [68, 28]] | (0.01, 0.5, 1, 50)                 |
| 7    | 0.7840    | [[592, 22], [72, 24]] | (0.01, 0.2, 1, 50)                 |
| 8    | 0.8118    | [[595, 7], [70, 37]]  | (0.001, 0.5, 2, 100)               |
| 9    | 0.8072    | [[607, 17], [69, 16]] | (0.01, 0.5, 1, 100)                |
| 10   | 0.7993    | [[610, 18], [61, 20]] | (0.001, 0.5, 1, 100)               |

### Heart Attack

The Heart Attack dataset showed a wide range of AUC scores, with the model generally struggling to detect the positive class effectively.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.5170    | [[674, 0], [36, 0]]   | (0.001, 0.2, 1, 50)                |
| 2    | 0.6429    | [[666, 0], [44, 0]]   | (0.01, 0.5, 1, 50)                 |
| 3    | 0.5957    | [[668, 0], [42, 0]]   | (0.001, 0.5, 2, 50)                |
| 4    | 0.4982    | [[675, 0], [35, 0]]   | (0.001, 0.2, 2, 100)               |
| 5    | 0.6438    | [[667, 0], [43, 0]]   | (0.01, 0.2, 1, 50)                 |
| 6    | 0.6044    | [[662, 0], [48, 0]]   | (0.01, 0.2, 1, 50)                 |
| 7    | 0.6030    | [[661, 0], [49, 0]]   | (0.01, 0.5, 1, 50)                 |
| 8    | 0.7008    | [[677, 0], [32, 0]]   | (0.001, 0.5, 2, 50)                |
| 9    | 0.6391    | [[679, 0], [30, 0]]   | (0.001, 0.5, 1, 100)               |
| 10   | 0.6452    | [[667, 0], [42, 0]]   | (0.01, 0.5, 2, 50)                 |

### High Blood Pressure

The High Blood Pressure dataset yielded relatively high AUC scores, suggesting effective class separation, although there were still significant numbers of false positives and negatives in the confusion matrices.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.7600    | [[326, 60], [135, 118]]| (0.01, 0.5, 1, 100)                |
| 2    | 0.7493    | [[313, 70], [126, 130]]| (0.01, 0.2, 2, 50)                 |
| 3    | 0.7573    | [[310, 68], [128, 133]]| (0.01, 0.5, 2, 50)                 |
| 4    | 0.7124    | [[308, 89], [120, 122]]| (0.001, 0.2, 2, 50)                |
| 5    | 0.7304    | [[310, 66], [136, 127]]| (0.001, 0.2, 1, 50)                |
| 6    | 0.7448    | [[290, 70], [134, 145]]| (0.01, 0.2, 1, 100)                |
| 7    | 0.7421    | [[322, 69], [131, 117]]| (0.01, 0.2, 1, 100)                |
| 8    | 0.7477    | [[321, 68], [132, 118]]| (0.001, 0.5, 1, 100)               |
| 9    | 0.7631    | [[312, 71], [124, 132]]| (0.001, 0.5, 1, 100)               |
| 10   | 0.7272    | [[308, 65], [139, 127]]| (0.01, 0.2, 2, 50)                 |

### Osteoporosis

The results for the Osteoporosis dataset indicate that the model struggled to achieve high AUC scores, and the confusion matrices suggest poor performance in detecting the positive class.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.5634    | [[644, 0], [66, 0]]   | (0.01, 0.5, 2, 100)                |
| 2    | 0.6499    | [[624, 4], [81, 1]]   | (0.001, 0.5, 1, 100)               |
| 3    | 0.6377    | [[639, 0], [71, 0]]   | (0.001, 0.5, 1, 50)                |
| 4    | 0.6677    | [[639, 0], [71, 0]]   | (0.01, 0.5, 2, 50)                 |
| 5    | 0.6014    | [[652, 0], [58, 0]]   | (0.01, 0.5, 1, 100)                |
| 6    | 0.6710    | [[647, 0], [63, 0]]   | (0.001, 0.5, 2, 50)                |
| 7    | 0.5889    | [[654, 0], [56, 0]]   | (0.001, 0.2, 1, 50)                |
| 8    | 0.4003    | [[658, 0], [51, 0]]   | (0.01, 0.2, 2, 100)                |
| 9    | 0.6391    | [[638, 0], [71, 0]]   | (0.001, 0.5, 1, 100)               |
| 10   | 0.6396    | [[644, 0], [65, 0]]   | (0.001, 0.5, 1, 50)                |

### Parkinsons

The Parkinsons dataset showed a wide range of AUC scores, with the model generally struggling to detect the positive class effectively.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.4172    | [[703, 0], [7, 0]]    | (0.01, 0.2, 2, 50)                 |
| 2    | 0.6811    | [[703, 0], [7, 0]]    | (0.001, 0.2, 2, 50)                |
| 3    | 0.3967    | [[639, 0], [71, 0]]   | (0.01, 0.5, 2, 100)                |
| 4    | 0.5055    | [[699, 0], [11, 0]]   | (0.001, 0.2, 1, 50)                |
| 5    | 0.5789    | [[699, 0], [11, 0]]   | (0.01, 0.5, 1, 100)                |
| 6    | 0.5122    | [[699, 0], [11, 0]]   | (0.001, 0.2, 1, 100)               |
| 7    | 0.7775    | [[707, 0], [3, 0]]    | (0.01, 0.2, 2, 100)                |
| 8    | 0.5642    | [[705, 0], [4, 0]]    | (0.001, 0.5, 2, 100)               |
| 9    | 0.4229    | [[707, 0], [2, 0]]    | (0.001, 0.5, 1, 100)               |
| 10   | 0.4229    | [[707, 0], [2, 0]]    | (0.01, 0.5, 2, 100)                |

### Stroke

The Stroke dataset showed some variability in AUC scores, with the model generally struggling to detect the positive class effectively.

| Fold | AUC Score | Confusion Matrix      | Hyper Parameters                   |
|------|-----------|-----------------------|-------------------------------------|
| 1    | 0.6140    | [[668, 0], [42, 0]]   | (0.01, 0.2, 1, 100)                |
| 2    | 0.6201    | [[660, 0], [50, 0]]   | (0.01, 0.2, 2, 50)                 |
| 3    | 0.6844    | [[662, 0], [48, 0]]   | (0.01, 0.2, 2, 50)                 |
| 4    | 0.6611    | [[668, 0], [42, 0]]   | (0.001, 0.2, 2, 100)               |
| 5    | 0.6532    | [[667, 0], [43, 0]]   | (0.001, 0.2, 1, 50)                |
| 6    | 0.7005    | [[680, 0], [30, 0]]   | (0.01, 0.5, 2, 50)                 |
| 7    | 0.6739    | [[671, 0], [39, 0]]   | (0.01, 0.5, 1, 50)                 |
| 8    | 0.6404    | [[664, 0], [45, 0]]   | (0.01, 0.5, 2, 50)                 |
| 9    | 0.7161    | [[669, 0], [40, 0]]   | (0.001, 0.5, 1, 100)               |
| 10   | 0.5186    | [[667, 0], [42, 0]]   | (0.001, 0.5, 1, 100)               |