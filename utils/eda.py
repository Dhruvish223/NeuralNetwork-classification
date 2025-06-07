import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df):
    with st.expander("📌 Column Types"):
        st.write(df.dtypes)

    with st.expander("📊 Distribution Plots"):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col = st.selectbox("Choose column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    with st.expander("📉 Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
