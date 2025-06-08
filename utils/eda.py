import streamlit as st
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from utils.gemini_api import get_gemini_summary

def run_eda(df, target):
    st.subheader("📌 Missing / Zero-Variance Features")
    stats = pd.DataFrame({
        'null_pct': (df.isnull().sum()/len(df)) * 100,
        'unique_count': df.nunique(),
        'dtype': df.dtypes
    })
    st.dataframe(stats)

    if st.checkbox("🔍 LLM Suggest Columns to Drop"):
        prompt = f"From these stats, suggest columns to drop for redundancy or leakage:\n\n{stats.to_string()}"
        suggestions = get_gemini_summary(prompt)
        st.text_area("LLM Suggestions", suggestions, height=200)
        to_drop = st.multiselect("Remove these columns", options=list(df.columns))
        df = df.drop(columns=to_drop) if to_drop else df

    st.subheader("📈 Univariate Distribution")
    num_cols = df.select_dtypes(['int64','float64']).columns
    cat_cols = df.select_dtypes('object').columns
    col = st.selectbox("Choose numeric column", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🔄 Bivariate Analysis")
    x = st.selectbox("X-axis (numeric)", num_cols)
    y = st.selectbox("Y-axis (numeric)", num_cols, index=1)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=target)
    st.pyplot(fig2)

    if st.checkbox("📊 LLM Suggest Insights on Bivariate Relationship"):
        prompt2 = f"Analyze correlation between {x} and {y} for target {target}:\n{df[[x,y,target]].corr().to_string()}"
        st.text_area("LLM Bivariate Analysis", get_gemini_summary(prompt2), height=200)

    return df.drop(target, axis=1), df[target]
