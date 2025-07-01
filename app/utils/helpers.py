"""
Helper functions for the application.
This module contains utility functions that can be used across the application.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

def sample_helper_function(x: int) -> int:
    """
    Example helper function that squares a number.
    
    Args:
        x: The number to square
        
    Returns:
        The square of the input number
    """
    logger.debug(f"Computing square of {x}")
    return x * x

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process a pandas DataFrame with common operations.
    
    Args:
        data: Input DataFrame to process
        
    Returns:
        Processed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Log the shape after processing
    logger.info(f"Processed data shape: {df.shape}")
    
    return df

def calculate_statistics(data: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
    """
    Calculate basic statistics for specified columns.
    
    Args:
        data: Input DataFrame
        columns: List of column names to analyze (if None, use all numeric columns)
        
    Returns:
        Dictionary with statistics for each column
    """
    if columns is None:
        # Get numeric columns
        columns = data.select_dtypes(include=np.number).columns.tolist()
    
    # Calculate statistics
    stats = {}
    for col in columns:
        if col in data.columns:
            col_stats = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
            }
            stats[col] = col_stats
    
    return stats

def format_currency(value: Union[int, float]) -> str:
    """
    Format a number as currency.
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string with $ prefix and commas
    """
    return f"${value:,.2f}"

def create_id(prefix: str = 'id') -> str:
    """
    Create a unique ID with a timestamp.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Unique ID string
    """
    import uuid
    import time
    
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    
    return f"{prefix}_{timestamp}_{unique_id}"