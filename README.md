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
├── data/raw/          # original dataset (not committed)
├── models/            # trained model, e.g. churn_pipeline.joblib (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                    # exploratory data analysis
│   ├── 02_baseline_models.ipynb        # baseline LogReg + Random Forest
│   └── 03_tuning_and_evaluation.ipynb  # tuning, evaluation, final model
├── src/
│   ├── data_loader.py     # load the raw CSV
│   ├── preprocessing.py   # cleaning, encoding, splitting
│   └── model.py           # deployable pipeline (preprocessing + classifier)
├── app/
│   └── streamlit_app.py   # interactive churn predictor
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

## Results
- Best model: Logistic Regression (tuned `C`, `class_weight="balanced"`) inside a
  preprocessing pipeline that accepts raw customer data.
- Test ROC-AUC: **~0.84**.
- Churn-class recall improved from **0.51** (baseline) to **~0.78** by weighting the
  minority class, the model now catches most churners, accepting more false alarms
  in exchange (the right tradeoff for a retention use case).

## Run the app
First create the model by running `notebooks/03_tuning_and_evaluation.ipynb`, then:
```powershell
streamlit run app/streamlit_app.py
```
Enter a customer's details and get a live churn probability.

## Progress
- [x] Day 1: project setup, data loading, exploratory data analysis
- [x] Day 2: preprocessing, feature engineering, baseline models
- [x] Day 3: model tuning, evaluation, Streamlit app
