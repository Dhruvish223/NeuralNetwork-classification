import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
from dotenv import load_dotenv
import time
from PIL import Image

# Import other modules
from data_processor import DataProcessor
from eda import EDA
from visualizations import Visualizer
from model import NeuralNetworkModel
from gemini_integration import GeminiLLM
import utils

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Global Superstore Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css():
    css = """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# Create session states
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'eda' not in st.session_state:
    st.session_state.eda = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data Upload & Cleaning", "Exploratory Data Analysis", 
         "Data Visualization", "Feature Engineering", "Model Training", 
         "Gemini LLM Insights", "About"]
selection = st.sidebar.radio("Go to", pages)

# Helper function to display dataset info
def display_dataset_info(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
    with col2:
        st.write(f"**Missing Values:** {df.isna().sum().sum()}")
        st.write(f"**Duplicates:** {df.duplicated().sum()}")
    
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Types")
    dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    dtype_df.index.name = "Column"
    dtype_df.reset_index(inplace=True)
    st.dataframe(dtype_df, use_container_width=True)

# Home page
if selection == "Home":
    st.title("Global Superstore Data Analysis Dashboard")
    st.markdown("""
    ## Welcome to the Global Superstore Analysis Tool
    
    This interactive dashboard allows you to analyze the Global Superstore dataset using advanced data science techniques. You can:
    
    - Clean and preprocess data with industry-standard methods
    - Explore data through comprehensive EDA
    - Visualize patterns and trends with interactive charts
    - Apply feature engineering techniques
    - Train neural network models for multiclass classification
    - Get AI-powered insights via Google's Gemini LLM
    
    ### Getting Started
    1. Navigate to the **Data Upload & Cleaning** section
    2. Upload your dataset or use the sample data
    3. Follow the steps in each section to analyze your data
    
    This tool is designed to be both powerful and user-friendly, suitable for both beginners and advanced users.
    """)
    
    # Display a sample image or dashboard preview
    st.image("https://via.placeholder.com/800x400?text=Global+Superstore+Dashboard", caption="Dashboard Preview")

# Data Upload & Cleaning
elif selection == "Data Upload & Cleaning":
    st.title("Data Upload & Cleaning")
    
    # File upload section
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    use_sample = st.checkbox("Use sample Global Superstore data")
    
    if use_sample:
        try:
            # Use sample data function from utils
            df = utils.get_sample_data()
            st.success("Sample data loaded successfully!")
            st.session_state.data = df
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    elif uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            st.session_state.data = df
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display dataset if loaded
    if st.session_state.data is not None:
        st.subheader("Dataset Overview")
        display_dataset_info(st.session_state.data)
        
        # Initialize the data processor
        if st.session_state.processor is None:
            st.session_state.processor = DataProcessor(st.session_state.data)
        
        # Data cleaning options
        st.subheader("Data Cleaning Options")
        
        clean_options = st.multiselect(
            "Select cleaning operations",
            ["Remove duplicates", "Handle missing values", "Fix data types", 
             "Remove outliers", "Normalize text fields", "Date formatting"]
        )
        
        if st.button("Apply Cleaning Operations"):
            with st.spinner("Cleaning data..."):
                # Process each selected cleaning option
                if "Remove duplicates" in clean_options:
                    st.session_state.processor.remove_duplicates()
                
                if "Handle missing values" in clean_options:
                    imputation_method = st.radio(
                        "Choose imputation method",
                        ["Mean/Mode", "Median", "KNN Imputer", "Drop rows", "Drop columns"]
                    )
                    st.session_state.processor.handle_missing_values(method=imputation_method)
                
                if "Fix data types" in clean_options:
                    st.session_state.processor.fix_data_types()
                
                if "Remove outliers" in clean_options:
                    outlier_method = st.radio(
                        "Choose outlier detection method",
                        ["IQR", "Z-Score", "Isolation Forest"]
                    )
                    st.session_state.processor.remove_outliers(method=outlier_method)
                
                if "Normalize text fields" in clean_options:
                    st.session_state.processor.normalize_text()
                
                if "Date formatting" in clean_options:
                    st.session_state.processor.format_dates()
                
                st.session_state.processed_data = st.session_state.processor.get_processed_data()
                st.success("Data cleaning completed!")
        
        # Display cleaned data if available
        if st.session_state.processed_data is not None:
            st.subheader("Cleaned Dataset")
            display_dataset_info(st.session_state.processed_data)
            
            # Option to download cleaned data
            st.download_button(
                label="Download cleaned data as CSV",
                data=st.session_state.processed_data.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

# Exploratory Data Analysis
elif selection == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data in the 'Data Upload & Cleaning' section first.")
    else:
        # Use cleaned data if available, otherwise use original data
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        # Initialize EDA class if not already done
        if st.session_state.eda is None:
            st.session_state.eda = EDA(df)
        
        # EDA options
        st.subheader("EDA Options")
        
        eda_options = st.multiselect(
            "Select EDA operations",
            ["Summary Statistics", "Correlation Analysis", "Distribution Analysis", 
             "Time Series Analysis", "Categorical Analysis", "Feature Importance"]
        )
        
        if "Summary Statistics" in eda_options:
            st.subheader("Summary Statistics")
            summary_stats = st.session_state.eda.get_summary_statistics()
            st.dataframe(summary_stats, use_container_width=True)
            
            st.subheader("Missing Value Analysis")
            missing_data = st.session_state.eda.analyze_missing_data()
            st.dataframe(missing_data, use_container_width=True)
        
        if "Correlation Analysis" in eda_options:
            st.subheader("Correlation Analysis")
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_cols = st.multiselect("Select columns for correlation analysis", numeric_cols, default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols)
            
            if selected_cols:
                correlation_plot = st.session_state.eda.plot_correlation(selected_cols)
                st.plotly_chart(correlation_plot, use_container_width=True)
            else:
                st.info("Please select columns for correlation analysis")
        
        if "Distribution Analysis" in eda_options:
            st.subheader("Distribution Analysis")
            
            # Select columns for distribution analysis
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            dist_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            dist_plot = st.session_state.eda.plot_distribution(dist_col)
            st.plotly_chart(dist_plot, use_container_width=True)
            
            # Add box plots for numeric columns
            st.subheader("Box Plots")
            box_cols = st.multiselect("Select columns for box plots", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols)
            
            if box_cols:
                box_plot = st.session
