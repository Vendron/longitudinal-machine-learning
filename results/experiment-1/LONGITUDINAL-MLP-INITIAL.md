# Longitudinal MLP Initial Results

**Parameters Used:**
- **Input Size:** Dynamic based on dataset
- **Hidden Layer Size:** 128
- **Output Size:** 1
- **Dropout Rate:** 0.4
- **Epochs:** 100
- **Learning Rate:** 0.3

**Summary:**
- **Angina, Cataract, Dementia, Diabetes, Osteoporosis, Parkinsons, Stroke:** The model failed to detect positive cases, resulting in F1 scores of 0.0.
- **Arthritis:** Very low performance with an F1 score of 0.0583, indicating poor model effectiveness.
- **High Blood Pressure, Heart Attack:** Slightly better results, but still poor overall, with F1 scores of 0.0854 and 0.1000 respectively.

| Dataset Used        | F1 Score | Precision | Recall | AUC Score | Confusion Matrix           |
|---------------------|----------|-----------|--------|-----------|----------------------------|
| Angina              | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[688, 0], [22, 0]]         |
| Arthritis           | 0.0583   | 0.5901    | 0.0301 | 0.5138    | [[410, 1], [290, 9]]        |
| Cataract            | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[483, 0], [227, 0]]        |
| Dementia            | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[697, 0], [13, 0]]         |
| Diabetes            | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[621, 0], [89, 0]]         |
| High Blood Pressure | 0.0854   | 0.7000    | 0.0455 | 0.5153    | [[396, 6], [294, 14]]       |
| Heart Attack        | 0.1000   | 0.5000    | 0.0556 | 0.5263    | [[672, 2], [34, 2]]         |
| Osteoporosis        | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[644, 0], [66, 0]]         |
| Parkinsons          | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[703, 0], [7, 0]]          |
| Stroke              | 0.0000   | 0.0000    | 0.0000 | 0.5000    | [[668, 0], [42, 0]]         |