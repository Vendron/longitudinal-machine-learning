# CNN Initial Results

**Parameters Used:**
- **Learning Rate:** 0.001
- **Max Epochs:** 100
- **Batch Size:** 32
- **Dropout Rate:** 0.5
- **CNN Channels:** 64
- **Kernel Size:** 3
- **Pool Size:** 2
- **Dense Units:** 1
- **Activation:** Sigmoid
- **Loss Function:** BCEWithLogitsLoss
- **Metrics:** Accuracy
- **Validation Split:** 0.1
- **Patience:** 10
- **Number of Layers:** 2
- **Test Size:** 0.1

**Summary:**
- **Arthritis, Diabetes:** The CNN model performed moderately well on these datasets, with F1 scores of 0.4238 and 0.4154, respectively.
- **Cataract, High Blood Pressure:** The model showed some capability in detecting positive cases, though with lower F1 scores.
- **Angina, Dementia, Heart Attack, Osteoporosis, Parkinsons, Stroke:** The model struggled significantly, failing to detect positive cases and resulting in F1 scores of 0.0.

The CNN model had mixed results, performing better on datasets with more balanced classes but struggling with highly imbalanced datasets.

| Dataset Used        | F1 Score | Precision | Recall | AUC Score | Confusion Matrix           |
|---------------------|----------|-----------|--------|-----------|----------------------------|
| Angina              | 0.0000   | 0.0000    | 0.0000 | 0.3131    | [[688, 0], [22, 0]]         |
| Arthritis           | 0.4238   | 0.6234    | 0.3211 | 0.6754    | [[353, 58], [203, 96]]      |
| Cataract            | 0.2572   | 0.4762    | 0.1762 | 0.6755    | [[439, 44], [187, 40]]      |
| Dementia            | 0.0000   | 0.0000    | 0.0000 | 0.5186    | [[697, 0], [13, 0]]         |
| Diabetes            | 0.4154   | 0.6585    | 0.3034 | 0.8193    | [[607, 14], [62, 27]]       |
| High Blood Pressure | 0.4606   | 0.6379    | 0.3604 | 0.7191    | [[339, 63], [197, 111]]     |
| Heart Attack        | 0.0000   | 0.0000    | 0.0000 | 0.5677    | [[673, 1], [36, 0]]         |
| Osteoporosis        | 0.0000   | 0.0000    | 0.0000 | 0.5088    | [[644, 0], [66, 0]]         |
| Parkinsons          | 0.0000   | 0.0000    | 0.0000 | 0.7149    | [[703, 0], [7, 0]]          |
| Stroke              | 0.0000   | 0.0000    | 0.0000 | 0.5493    | [[668, 0], [42, 0]]         |