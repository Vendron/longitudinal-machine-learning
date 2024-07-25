# Machine Learning for Longitudinal Medical Data Analysis
> University Masters Thesis Project
## Abstract
The aim of this project is to develop several deep learning models that can predict the onset of a disease based on longitudinal medical (ELSA) data. The model will be trained on a dataset of patients with a specific disease and will predict the onset of the disease in other patients over time.

The model will be evaluated on a separate test set to determine its accuracy and performance. The project will involve preprocessing the data, training the model, and evaluating the performance of each models with benchmarking.

The results of the project will be used to determine the feasibility of using deep learning for longitudinal medical data analysis.

## Setup
### Install Dependencies
```sh
pip install -r requirements.txt
```
### Setup Environment Variables
```sh
cp .env.example .env
```
### Run Project
```sh
python src/main.py
```
----
## Project Structure
```
├── data/
├── src/
│   ├── features/
│   │   └── preprocessing.py
│   ├── models/
│   │   └── mlp.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
├── tests/
│   └── test_classifier_mlp.py
├── .env
├── .env.sample
├── .gitignore
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── setup.sh
```