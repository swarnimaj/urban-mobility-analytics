"""
Tests for utility functions.
"""
import unittest
import pandas as pd
import geopandas as gpd
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import ensure_directory, save_dataframe, load_dataframe

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path("tests/temp_data")
        ensure_directory(self.test_dir)
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        for file in self.test_dir.glob("*"):
            file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_save_and_load_csv(self):
        """Test saving and loading a CSV file."""
        save_dataframe(self.test_df, "test_data", self.test_dir, "csv")
        loaded_df = load_dataframe("test_data", self.test_dir, "csv")
        
        self.assertTrue((self.test_df == loaded_df).all().all())
        self.assertTrue(Path(self.test_dir, "test_data.csv").exists())

if __name__ == '__main__':
    unittest.main()