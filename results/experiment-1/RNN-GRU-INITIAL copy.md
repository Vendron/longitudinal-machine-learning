# RNN-GRU Initial Results

**Parameters Used:**
- **Learning Rate:** 0.001
- **Max Epochs:** 100
- **Batch Size:** 32
- **Bidirectional:** True
- **Dropout Rate:** 0.5
- **GRU Units:** 100
- **Dense Units:** 1
- **Activation:** Sigmoid
- **Loss Function:** BCEWithLogitsLoss
- **Validation Split:** 0.1
- **Number of Layers:** 2
- **Test Size:** 0.1

**Summary:**
- **High Blood Pressure:** The GRU model performed best on this dataset, with an F1 score of 0.5461.
- **Arthritis, Diabetes, Cataract:** The model showed moderate performance, with F1 scores ranging from 0.3420 to 0.4658.
- **Angina, Dementia, Heart Attack, Osteoporosis, Parkinsons, Stroke:** The model struggled with these datasets, especially those with significant class imbalance, resulting in F1 scores of 0.0.

The GRU model exhibited similar strengths and weaknesses to the LSTM, performing relatively well on some datasets but failing on highly imbalanced ones.

| Dataset Used        | F1 Score | Precision | Recall | AUC Score | Confusion Matrix           |
|---------------------|----------|-----------|--------|-----------|----------------------------|
| Angina              | 0.0000   | 0.0000    | 0.0000 | 0.5806    | [[688, 0], [22, 0]]         |
| Arthritis           | 0.4658   | 0.6450    | 0.3645 | 0.6416    | [[351, 60], [190, 109]]     |
| Cataract            | 0.3420   | 0.2599    | 0.5000 | 0.6709    | [[424, 59], [168, 59]]      |
| Dementia            | 0.0000   | 0.0000    | 0.0000 | 0.6659    | [[697, 0], [13, 0]]         |
| Diabetes            | 0.4571   | 0.6275    | 0.3596 | 0.8078    | [[602, 19], [57, 32]]       |
| High Blood Pressure | 0.5461   | 0.6325    | 0.4805 | 0.7142    | [[316, 86], [160, 148]]     |
| Heart Attack        | 0.0000   | 0.0000    | 0.0000 | 0.6927    | [[674, 0], [36, 0]]         |
| Osteoporosis        | 0.0000   | 0.0000    | 0.0000 | 0.6210    | [[644, 0], [66, 0]]         |
| Parkinsons          | 0.0000   | 0.0000    | 0.0000 | 0.6921    | [[703, 0], [7, 0]]          |
| Stroke              | 0.0000   | 0.0000    | 0.0000 | 0.6584    | [[668, 0], [42, 0]]         |