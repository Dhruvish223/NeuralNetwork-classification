import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.gemini_api import get_gemini_summary

def evaluate_and_download(y_test, y_pred, history, report, le):
    st.subheader("📋 Classification Report")
    df = pd.DataFrame(report).transpose().style.format("{:.2f}")
    st.dataframe(df)

    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    if len(le.classes_) == 2:
        st.subheader("📊 ROC-AUC (Binary)")
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        probas = history.model.predict(history.model.validation_data[0]) if False else None

    st.subheader("📈 Training History")
    fig2, ax = plt.subplots(1, 2, figsize=(12,4))
    ax[0].plot(history.history['loss'], label='loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='acc')
    ax[1].plot(history.history['val_accuracy'], label='val_acc')
    ax[1].legend()
    st.pyplot(fig2)

    llm_summary = get_gemini_summary(str(report))
    st.subheader("💬 Gemini Report Summary")
    st.text_area("LLM Summary", llm_summary, height=200)

    content = "Final Report\\n"+str(pd.DataFrame(report).transpose().to_string())+"\\nGemini Insights:\\n"+llm_summary
    st.download_button("📥 Download Report", content, file_name="final_report.txt")
