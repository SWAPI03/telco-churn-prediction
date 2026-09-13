"""Streamlit app: enter a customer's details and get a churn probability.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make the project root importable so `from src.model import ...` works.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.model import load_model

st.set_page_config(page_title="Churn Predictor", page_icon="phone")
st.title("Telco Customer Churn Predictor")
st.write("Enter a customer's details to estimate their probability of churning.")


@st.cache_resource
def get_model():
    """Load the trained pipeline once and cache it across reruns."""
    return load_model()


model = get_model()

st.subheader("Customer details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior citizen", [0, 1])
    Partner = st.selectbox("Has partner", ["Yes", "No"])
    Dependents = st.selectbox("Has dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    OnlineSecurity = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

with col2:
    TechSupport = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )

    # ---- YOUR TASK: add the five remaining inputs, following the pattern above ----
    # Replace each None with the widget described in the comment.
    #   InternetService : st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    #   Contract        : st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    #   tenure          : st.slider("Tenure (months)", 0, 72, 12)
    #   MonthlyCharges  : st.slider("Monthly charges", 18.0, 120.0, 70.0)
    #   TotalCharges    : st.number_input("Total charges", 0.0, 9000.0, 1400.0)
    InternetService = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]) 
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.slider("Monthly charges", 18.0, 120.0, 70.0)
    TotalCharges = st.number_input("Total charges", 0.0, 9000.0, 1400.0)

# Assemble a one-row DataFrame with the SAME column names the model trained on.
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}])

if st.button("Predict churn"):
    proba = model.predict_proba(customer)[0, 1]
    st.metric("Churn probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.error("High risk: this customer is likely to churn.")
    else:
        st.success("Low risk: this customer is likely to stay.")
