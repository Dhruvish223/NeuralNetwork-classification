import streamlit as st
import pandas as pd
import os

from utils.cleaning import clean_data
from utils.eda import run_eda
from utils.feature_engineering import engineer_features
from utils.model import train_model
from utils.gemini_api import get_gemini_summary

st.set_page_config(page_title="Global Superstore Dashboard", layout="wide")

st.title("📊 Global Superstore Analysis Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    st.info("ℹ️ Please upload a CSV to continue.")
    st.stop()

# Cleaning
st.subheader("🔧 Data Cleaning")
df_clean = clean_data(df)
st.dataframe(df_clean.head())

# EDA
st.subheader("📈 Exploratory Data Analysis")
run_eda(df_clean)

# Feature Engineering
st.subheader("⚙️ Feature Engineering")
X, y, label_encoder = engineer_features(df_clean)
st.write("Features and target prepared for modeling.")

# Modeling
st.subheader("🧠 Train Neural Network")
accuracy, report = train_model(X, y)
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:\n" + report)

# Gemini Integration
st.subheader("💬 AI-Powered Summary & Suggestions")
if st.button("Get Summary from Gemini"):
    summary = get_gemini_summary(report)
    st.markdown(summary)
