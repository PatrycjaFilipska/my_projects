
Shelter Dogs — EDA & Machine Learning Project

Overview

This project explores a dataset of shelter dogs and builds a machine-learning model that predicts whether a dog is good with children based on various characteristics such as breed, size, age, energy level, coat color and more.

The project includes:

Exploratory Data Analysis (EDA)

Data cleaning and preprocessing

Feature engineering

Machine-learning model (Random Forest)

Experiment 1 - model comparison

Model evaluation (classification report, confusion matrix, feature importance)

Example inference on a new dog profile

Dataset

The dataset comes from Kaggle:
Shelter Dogs Dataset
🔗 https://www.kaggle.com/datasets/\
<your-dataset-link>

It contains information on dogs available for adoption, including:

breed

sex

size

age

coat color

energy level

behavior labels

adoption-related features

Repository Structure

```text
shelter-dogs-ml/
│
├── notebooks/
│   └── Shelter_Dogs.ipynb       # main notebook: EDA + models
│
├── data/
│   └── dogs.csv                 # (optional) dataset or a sample
│
├── requirements.txt             # Python dependencies
└── README.md

Model

The final model is a RandomForestClassifier with class balancing:

RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


Why RandomForest?

handles categorical features well (after encoding)

robust to outliers

performs well without heavy hyperparameter tuning


