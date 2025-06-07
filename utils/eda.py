import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df):
    with st.expander("📌 Column Types"):
        st.write(df.dtypes)

    with st.expander("📊 Distribution Plots"):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Choose column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for distribution plots.")

    with st.expander("📉 Correlation Heatmap"):
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns to compute correlation.")

