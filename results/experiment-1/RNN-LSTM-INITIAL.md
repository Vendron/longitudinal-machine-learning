# RNN-LSTM Initial Results

**Parameters Used:**
- **Learning Rate:** 0.001
- **Max Epochs:** 100
- **Batch Size:** 32
- **Bidirectional:** True
- **Dropout Rate:** 0.5
- **LSTM Units:** 100
- **Dense Units:** 1
- **Activation:** Sigmoid
- **Loss Function:** BCEWithLogitsLoss
- **Metrics:** Accuracy
- **Validation Split:** 0.1
- **Patience:** 10
- **Number of Layers:** 2
- **Test Size:** 0.1

**Summary:**
- **High Blood Pressure:** The model performed best on this dataset, with an F1 score of 0.5135.
- **Angina, Cataract, Diabetes, Osteoporosis, Stroke:** The model showed moderate performance with F1 scores ranging from 0.3099 to 0.4703.
- **Arthritis, Dementia, Heart Attack, Parkinsons:** The model struggled with these datasets, especially those with significant class imbalance, resulting in F1 scores of 0.0. 

The AUC scores indicate that while the LSTM could differentiate between classes to some extent, it struggled with binary classification, particularly in imbalanced datasets.

| Dataset Used        | F1 Score | AUC Score | Confusion Matrix           |
|---------------------|----------|-----------|----------------------------|
| Angina              | 0.4703   | 0.6588    | [[349, 62], [188, 111]]     |
| Arthritis           | 0.0000   | 0.6110    | [[688, 0], [22, 0]]         |
| Cataract            | 0.3099   | 0.6596    | [[421, 62], [174, 53]]      |
| Dementia            | 0.0000   | 0.7218    | [[697, 0], [13, 0]]         |
| Diabetes            | 0.4460   | 0.8136    | [[602, 19], [58, 31]]       |
| High Blood Pressure | 0.5135   | 0.7070    | [[325, 77], [175, 133]]     |
| Heart Attack        | 0.0000   | 0.6735    | [[674, 0], [36, 0]]         |
| Osteoporosis        | 0.4703   | 0.6588    | [[349, 62], [188, 111]]     |
| Parkinsons          | 0.0000   | 0.6110    | [[688, 0], [22, 0]]         |
| Stroke              | 0.3099   | 0.6596    | [[421, 62], [174, 53]]      |