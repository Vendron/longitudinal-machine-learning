# Traditional MLP Initial Results

**Parameters Used:**
- **Hidden Layer Size:** 128
- **Dropout Rate:** 0.4
- **Learning Rate:** 0.3
- **Epochs:** 100
- **Output Size:** 1

**Summary:**
- **Angina, Cataract, Dementia, Heart Attack, Osteoporosis, Parkinsons, Stroke:** The model failed to detect positive cases, resulting in F1 scores of 0.0.
- **Arthritis:** Moderate performance with an F1 score of 0.6324.
- **Diabetes:** Low F1 score (0.1458), with high precision but poor recall.

| Dataset Used        | F1 Score | Precision | Recall | AUC Score | Confusion Matrix           |
|---------------------|----------|-----------|--------|-----------|----------------------------|
| Angina              | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[688, 0], [22, 0]]         |
| Arthritis           | 0.6324   | 0.5742    | 0.4916 | 0.6132    | [[302, 109], [152, 147]]    |
| Cataract            | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[483, 0], [227, 0]]        |
| Dementia            | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[697, 0], [13, 0]]         |
| Diabetes            | 0.1458   | 1.0000    | 0.0787 | 0.5000    | [[621, 0], [82, 7]]         |
| High Blood Pressure | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[402, 0], [308, 0]]        |
| Heart Attack        | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[674, 0], [36, 0]]         |
| Osteoporosis        | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[644, 0], [66, 0]]         |
| Parkinsons          | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[703, 0], [7, 0]]          |
| Stroke              | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[668, 0], [42, 0]]         |