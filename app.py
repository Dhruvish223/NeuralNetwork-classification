import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
import google.generativeai as genai

# -------- SETUP --------
st.set_page_config(layout="wide")
st.title("📊 Global Superstore - EDA, Visualization, and AI Modeling Dashboard")

# Gemini API Key input
api_key = st.sidebar.text_input("Enter your Gemini API key:", type="password")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")

def gemini_insight(prompt):
    if api_key:
        response = model.generate_content(prompt)
        return response.text
    return "[Gemini not active: No API key provided.]"

# -------- FILE UPLOAD --------
file = st.file_uploader("Upload the Global Superstore Excel File", type=["xlsx"])

if file:
    df = pd.read_excel(file, sheet_name="Orders")
    df.columns = df.columns.str.strip()

    with st.expander("📄 Raw Data Preview"):
        st.dataframe(df.head())

    # Data cleaning
    df = df[df['Sales'] != 0].copy()
    df['Percent Profit'] = df['Profit'] / df['Sales'] * 100
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Month'] = df['Order Date'].dt.to_period("M").astype(str)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Categorical Viz", "📈 Numeric Viz", "⏳ Time Series", "🧠 Neural Network"])

    with tab1:
        st.header("Categorical Visualizations")
        for col in ["Segment", "Region", "Category", "Ship Mode", "Order Priority"]:
            fig = px.histogram(df, x=col, color="Order Priority", barmode="group",
                               title=f"{col} vs Order Priority")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Numerical and Correlation Visualizations")

        # Box plots
        st.subheader("Box Plots by Segment")
        fig = px.box(df, x="Segment", y="Percent Profit", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        st.subheader("Sales vs Profit Scatter")
        fig = px.scatter(df, x="Sales", y="Profit", color="Order Priority", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric = df.select_dtypes(include=np.number)
        corr = numeric.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.header("Time Series Analysis")
        fig = px.histogram(df, x="Order Month", color="Order Priority", title="Order Priority Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Neural Network Classification")

        # Label encode target
        df_nn = df.copy()
        le = LabelEncoder()
        df_nn['Order Priority'] = le.fit_transform(df_nn['Order Priority'])

        st.subheader("Option A: Basic Neural Network")
        features = ["Sales", "Profit", "Discount", "Quantity"]
        X = df_nn[features]
        y = df_nn['Order Priority']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model_basic = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])

        model_basic.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history_basic = model_basic.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
        y_pred_basic = np.argmax(model_basic.predict(X_test), axis=1)

        st.text("Classification Report - Basic Model")
        st.text(classification_report(y_test, y_pred_basic, target_names=le.classes_))

        st.subheader("Option B: Tuned & Enhanced Neural Network")
        # Feature engineering
        df_fe = df_nn.copy()
        df_fe = pd.get_dummies(df_fe, columns=['Segment', 'Region', 'Category', 'Ship Mode'], drop_first=True)
        X2 = df_fe.drop(['Order ID', 'Customer ID', 'Customer Name', 'Order Date', 'Ship Date',
                         'Order Priority', 'Product Name', 'Product ID', 'Country', 'City', 'State', 'Postal Code'], axis=1)

        y2 = df_fe['Order Priority']
        scaler2 = StandardScaler()
        X2_scaled = scaler2.fit_transform(X2)

        # Optional PCA
        pca = PCA(n_components=0.95)
        X2_pca = pca.fit_transform(X2_scaled)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_pca, y2, test_size=0.3, random_state=42)

        def build_model(hp):
            model = keras.Sequential()
            model.add(layers.InputLayer(input_shape=(X_train2.shape[1],)))
            for i in range(hp.Int("layers", 1, 3)):
                model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
                                       activation=hp.Choice("act", ["relu", "tanh"])))
            model.add(layers.Dense(4, activation='softmax'))
            model.compile(optimizer=keras.optimizers.Adam(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5, overwrite=True,
                                directory='my_dir', project_name='superstore')
        tuner.search(X_train2, y_train2, epochs=10, validation_split=0.2, verbose=0)
        best_model = tuner.get_best_models(1)[0]
        y_pred_advanced = np.argmax(best_model.predict(X_test2), axis=1)

        st.text("Classification Report - Tuned Model")
        st.text(classification_report(y_test2, y_pred_advanced, target_names=le.classes_))

        # Comparison
        st.subheader("Model Accuracy Comparison")
        acc1 = accuracy_score(y_test, y_pred_basic)
        acc2 = accuracy_score(y_test2, y_pred_advanced)
        st.metric("Basic Model Accuracy", f"{acc1:.2%}")
        st.metric("Tuned Model Accuracy", f"{acc2:.2%}")

        if api_key:
            st.subheader("🤖 Gemini Insights")
            suggestion = gemini_insight(
                f"Here are two classification reports for predicting Order Priority using neural networks.\n\n"
                f"Basic Model:\n{classification_report(y_test, y_pred_basic)}\n\n"
                f"Tuned Model:\n{classification_report(y_test2, y_pred_advanced)}\n\n"
                f"Suggest improvements or business insights.")
            st.markdown(suggestion)

else:
    st.info("📤 Please upload the Global Superstore Excel file to begin.")
