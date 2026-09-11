# Telco Customer Churn Prediction

Predict which telecom customers are likely to leave (churn), using classical
machine learning on the IBM "Telco Customer Churn" dataset (7,043 customers,
21 features).

## Goals
- Explore the drivers of customer churn (EDA).
- Build and compare several classification models.
- Evaluate with metrics that matter for imbalanced data (precision, recall, ROC-AUC).
- Serve the best model through a small interactive Streamlit app.

## Project structure
```
telco-churn-prediction/
├── data/
│   ├── raw/          # original dataset (not committed)
│   └── processed/    # cleaned data (not committed)
├── models/           # trained model files (not committed)
├── notebooks/
│   └── 01_eda.ipynb  # exploratory data analysis
├── src/
│   ├── data_loader.py
│   └── ...
├── requirements.txt
└── README.md
```

## Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset
IBM sample "Telco Customer Churn" dataset. Download it from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
rename the CSV to `telco_churn.csv`, and place it in `data/raw/`.

## Progress
- [x] Day 1: project setup, data loading, exploratory data analysis
- [ ] Day 2: preprocessing, feature engineering, baseline models
- [ ] Day 3: model tuning, evaluation, Streamlit app
