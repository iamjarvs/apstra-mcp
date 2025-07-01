#!/usr/bin/env python3
"""
Main entry point for the application.
This file serves as the bridge between the backend logic and the Streamlit UI.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add the app directory to the path so we can import modules
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Import UI components
from ui.streamlit_app import run_streamlit_app

def main():
    """Main function to initialize and run the application."""
    logger.info("Starting application")
    
    # Initialize any services, databases, or connections here
    # example: initialize_database()
    
    # Run the Streamlit app
    run_streamlit_app()

if __name__ == "__main__":
    main()