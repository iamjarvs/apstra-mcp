"""
Test cases for the application.
This file contains test functions for the application modules.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the app directory to the path for imports
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Import the modules to test
from app.utils.helpers import sample_helper_function, format_currency, create_id

def test_sample_helper_function():
    """Test the sample helper function."""
    # Test with a positive number
    result = sample_helper_function(5)
    assert result == 25, f"Expected 25, got {result}"
    
    # Test with zero
    result = sample_helper_function(0)
    assert result == 0, f"Expected 0, got {result}"
    
    # Test with a negative number
    result = sample_helper_function(-3)
    assert result == 9, f"Expected 9, got {result}"

def test_format_currency():
    """Test the currency formatting function."""
    # Test with an integer
    result = format_currency(1000)
    assert result == "$1,000.00", f"Expected '$1,000.00', got {result}"
    
    # Test with a float
    result = format_currency(1234.56)
    assert result == "$1,234.56", f"Expected '$1,234.56', got {result}"
    
    # Test with zero
    result = format_currency(0)
    assert result == "$0.00", f"Expected '$0.00', got {result}"

def test_create_id():
    """Test the ID creation function."""
    # Test with default prefix
    id1 = create_id()
    assert id1.startswith("id_"), f"ID should start with 'id_', got {id1}"
    
    # Test with custom prefix
    id2 = create_id("test")
    assert id2.startswith("test_"), f"ID should start with 'test_', got {id2}"
    
    # Test uniqueness
    id3 = create_id()
    assert id1 != id3, f"IDs should be unique, got {id1} and {id3}"

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-v", __file__])