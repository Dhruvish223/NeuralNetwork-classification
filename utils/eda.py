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

    # --- 1. NULL VALUE ANALYSIS ---
    with st.expander("🧼 Missing Values Summary"):
        null_counts = df.isnull().sum()
        null_percent = (df.isnull().sum() / len(df)) * 100
        null_df = pd.DataFrame({
            'Missing Count': null_counts,
            'Missing %': null_percent.round(2)
        }).sort_values(by="Missing %", ascending=False)

        st.dataframe(null_df.style.background_gradient(cmap='Reds'))
    
    # --- 2. OUTLIER DETECTION USING BOXPLOTS ---
    with st.expander("📦 Outlier Detection (Box & Whiskers)"):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_cols = st.multiselect("Select columns to analyze outliers:", numeric_cols, default=numeric_cols)

        if selected_cols:
            for col in selected_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

    # --- 3. GEMINI ANALYSIS OF OUTLIERS ---
    with st.expander("🤖 Gemini Evaluation of Outliers"):
        if st.button("🔍 Analyze Outliers using Gemini"):
            outlier_summary = []

            for col in selected_cols:
                desc = df[col].describe()
                iqr = desc["75%"] - desc["25%"]
                lower = desc["25%"] - 1.5 * iqr
                upper = desc["75%"] + 1.5 * iqr
                outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]

                outlier_summary.append(f"Column: **{col}**\n- Outliers Detected: {outlier_count} values outside [{lower:.2f}, {upper:.2f}]")

            prompt = "Analyze the following outlier detection summary and suggest treatments if needed:\n\n" + "\n\n".join(outlier_summary)
            analysis = get_gemini_summary(prompt)
            st.markdown(analysis)

