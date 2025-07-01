"""
Streamlit UI module.
This file contains all the Streamlit UI components and logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import utils
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import utility functions
from utils.helpers import sample_helper_function

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Your App Name",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def sidebar():
    """Create and handle the sidebar elements."""
    st.sidebar.title("Navigation")
    
    # Add sidebar navigation options
    page = st.sidebar.radio("Select a page:", ["Home", "Data Analysis", "About"])
    
    # Add other sidebar elements like filters, parameters, etc.
    st.sidebar.header("Settings")
    
    # Example parameters
    parameter1 = st.sidebar.slider("Parameter 1", 0, 100, 50)
    parameter2 = st.sidebar.selectbox("Parameter 2", ["Option 1", "Option 2", "Option 3"])
    
    return page, parameter1, parameter2

def home_page():
    """Render the home page."""
    st.title("Welcome to Your Application")
    st.write("This is a template for your Streamlit applications.")
    
    # Example of using the helper function
    result = sample_helper_function(10)
    st.write(f"Example calculation result: {result}")
    
    # Example data visualization
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(chart_data)

def data_analysis_page():
    """Render the data analysis page."""
    st.title("Data Analysis")
    st.write("This page is for data analysis and visualization.")
    
    # Example file uploader
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Attempt to read the file as CSV
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # More data analysis would go here
            st.write("Data Statistics:")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def about_page():
    """Render the about page."""
    st.title("About This Application")
    st.write("""
    This is a template application created to streamline the development of 
    Streamlit-based Python applications.
    
    Key features:
    - Modular code structure
    - Docker deployment
    - GitHub Actions integration
    - Basic UI components
    """)

def run_streamlit_app():
    """Main function to run the Streamlit application."""
    logger.info("Starting Streamlit UI")
    
    # Set up the page configuration
    setup_page_config()
    
    # Create the sidebar and get user selections
    page, param1, param2 = sidebar()
    
    # Render the appropriate page based on user selection
    if page == "Home":
        home_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "About":
        about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("### Created with the Python Streamlit Template")

if __name__ == "__main__":
    run_streamlit_app()