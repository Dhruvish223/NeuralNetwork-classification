import streamlit as st
import pandas as pd
from utils.data_cleaning import clean_data
from utils.eda import run_eda
from utils.feature_engineering import engineer_features
from utils.model_training import tune_and_train_model
from utils.evaluation import evaluate_and_download
from utils.gemini_api import get_gemini_summary

st.set_page_config(layout='wide', page_title="Global Superstore AI-MLOps Dashboard")

st.title("📊 AI-Powered Superstore Analytics & Modeling")

uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx"])
if not uploaded_file:
    st.warning("Upload data to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
st.success("✅ Data Loaded")

df_cleaned = clean_data(df)

st.sidebar.subheader("Select Target Column")
target_column = st.sidebar.selectbox("Target", options=[None] + list(df_cleaned.columns))
if not target_column:
    st.warning("Please select a Target column")
    st.stop()

st.sidebar.subheader("Model Parameters")
params = {
    "max_trials": st.sidebar.number_input("Keras-Tuner Trials", 1, 20, 5),
    "epochs": st.sidebar.number_input("Max Epochs", 5, 100, 20),
    "layer1_units": st.sidebar.slider("Layer 1 units", 32, 256, 128),
    "layer2_units": st.sidebar.slider("Layer 2 units", 16, 128, 64),
    "dropout": st.sidebar.slider("Dropout rate", 0.0, 0.5, 0.3),
}

st.header("1️⃣ EDA & Feature Insights")
X_df, y_series = run_eda(df_cleaned, target_column)

st.header("2️⃣ Feature Engineering")
X, y, encoders = engineer_features(X_df, y_series)

st.header("3️⃣ Model Tuning & Training")
model, history, y_test, y_pred, report = tune_and_train_model(X, y, params)

st.header("4️⃣ Evaluation & Download")
evaluate_and_download(y_test, y_pred, history, report, encoders)

st.header("5️⃣ LLM Insights & Suggestions")
insights = get_gemini_summary(report)
st.text_area("Gemini Insights", insights, height=300)
