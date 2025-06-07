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
                box_plot = st.session)
    suffix : str
        Suffix to add (e.g., '%')
        
    Returns:
    --------
    str
        Formatted number
    """
    if abs(number) >= 1e9:
        return f"{prefix}{number/1e9:.2f}{suffix} B"
    elif abs(number) >= 1e6:
        return f"{prefix}{number/1e6:.2f}{suffix} M"
    elif abs(number) >= 1e3:
        return f"{prefix}{number/1e3:.2f}{suffix} K"
    else:
        return f"{prefix}{number:.2f}{suffix}"

def create_metrics_dashboard(metrics_dict, title="Key Metrics"):
    """
    Create a metrics dashboard
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names and values
    title : str
        Dashboard title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Dashboard figure
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=len(metrics_dict) // 2 + len(metrics_dict) % 2,
        cols=2,
        subplot_titles=list(metrics_dict.keys())
    )
    
    # Add metrics
    for i, (metric, value) in enumerate(metrics_dict.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Format value
        if isinstance(value, (int, float)):
            formatted_value = format_number(value)
        else:
            formatted_value = str(value)
        
        # Add indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=value if isinstance(value, (int, float)) else 0,
                number={'prefix': "", 'suffix': "", 'valueformat': ".2f"},
                title={'text': f"{metric}<br><span style='font-size:0.8em'>{formatted_value}</span>"}
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=300 * (len(metrics_dict) // 2 + len(metrics_dict) % 2),
        title_text=title,
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def get_sample_data():
    """
    Get sample Global Superstore data
    
    Returns:
    --------
    pandas.DataFrame
        Sample dataset
    """
    # This is a placeholder function
    # In a real implementation, you'd provide access to a sample dataset
    
    # Generate a simple sample dataset if the real one is not available
    np.random.seed(42)
    
    # Generate sample dates
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    days = (end_date - start_date).days
    dates = [start_date + datetime.timedelta(days=np.random.randint(0, days)) for _ in range(1000)]
    
    # Sample regions, countries, and categories
    regions = ['North', 'South', 'East', 'West', 'Central']
    countries = ['USA', 'UK', 'France', 'Germany', 'China', 'Japan', 'Brazil', 'India', 'Australia', 'Canada']
    categories = ['Furniture', 'Office Supplies', 'Technology']
    sub_categories = {
        'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
        'Office Supplies': ['Storage', 'Paper', 'Binders', 'Art', 'Supplies'],
        'Technology': ['Phones', 'Machines', 'Accessories', 'Copiers']
    }
    ship_modes = ['Standard Class', 'First Class', 'Second Class', 'Same Day']
    segments = ['Consumer', 'Corporate', 'Home Office']
    
    # Generate data
    data = {
        'Order Date': dates,
        'Ship Date': [date + datetime.timedelta(days=np.random.randint(1, 14)) for date in dates],
        'Region': [np.random.choice(regions) for _ in range(1000)],
        'Country': [np.random.choice(countries) for _ in range(1000)],
        'Category': [np.random.choice(categories) for _ in range(1000)],
        'Ship Mode': [np.random.choice(ship_modes) for _ in range(1000)],
        'Customer Segment': [np.random.choice(segments) for _ in range(1000)],
        'Sales': np.random.uniform(100, 5000, 1000),
        'Quantity': np.random.randint(1, 20, 1000),
        'Discount': np.random.uniform(0, 0.5, 1000),
        'Profit': np.random.uniform(-500, 2000, 1000),
        'Shipping Cost': np.random.uniform(10, 500, 1000)
    }
    
    # Add Sub-Category based on Category
    sub_cats = []
    for cat in data['Category']:
        sub_cats.append(np.random.choice(sub_categories[cat]))
    data['Sub-Category'] = sub_cats
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Add an ID column
    df['Order ID'] = ['ORD-' + str(i).zfill(5) for i in range(1, 1001)]
    
    # Add some missing values
    for col in df.columns:
        if col not in ['Order ID', 'Order Date', 'Ship Date']:
            mask = np.random.choice([True, False], size=len(df), p=[0.05, 0.95])
            df.loc[mask, col] = np.nan
    
    # Add some duplicated rows
    duplicate_indices = np.random.choice(df.index, size=50, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates])
    
    return df
