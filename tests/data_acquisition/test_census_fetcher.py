# File: tests/data_acquisition/test_census_fetcher.py
"""
Unit tests for the census_fetcher module.
"""
import unittest
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_acquisition.fetch_census_data import CensusFetcher

class TestCensusFetcher(unittest.TestCase):
    """Tests for the CensusFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a test API key from environment or a placeholder
        test_api_key = os.environ.get("CENSUS_API_KEY", None)
        self.fetcher = CensusFetcher(api_key=test_api_key, year=2021)
        
        # Test data
        self.test_state = "WA"
        self.test_county = "King"
    
    def test_state_fips_conversion(self):
        """Test conversion of state names to FIPS codes."""
        self.assertEqual(self.fetcher._get_state_fips("WA"), "53")
        self.assertEqual(self.fetcher._get_state_fips("Washington"), "53")
        self.assertEqual(self.fetcher._get_state_fips("53"), "53")
        
        with self.assertRaises(ValueError):
            self.fetcher._get_state_fips("Invalid State")
            
    def test_get_census_boundaries(self):
        """Test fetching census tract boundaries."""
        # Skip if shapefile doesn't exist
        tiger_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "census" / "tiger" / "tl_2021_53_tract"
        shp_file = tiger_dir / f"tl_2021_53_tract.shp"
        if not shp_file.exists():
            self.skipTest(f"Census tract shapefile not found at {shp_file}")
        
        boundaries = self.fetcher.get_census_boundaries(self.test_state, self.test_county)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(boundaries, gpd.GeoDataFrame)
        
        # Check that it has geometries
        self.assertTrue('geometry' in boundaries.columns)
        
        # Check that it has some rows
        self.assertTrue(len(boundaries) > 0)
        
        # Check that it has a geoid column
        self.assertTrue('geoid' in boundaries.columns)
        
        # Check that all tracts are in the correct county
        if 'COUNTYFP' in boundaries.columns:
            # King County, WA has FIPS code 033
            expected_county_fips = "033"
            self.assertTrue(all(boundaries['COUNTYFP'] == expected_county_fips))

    
    def test_get_demographic_data(self):
        """Test fetching demographic data."""
        # Skip if no API key available
        if not self.fetcher.api_key:
            self.skipTest("No Census API key available")
        
        demographics = self.fetcher.get_demographic_data(state=self.test_state, county=self.test_county)
        
        # Check that we got a DataFrame
        self.assertIsInstance(demographics, pd.DataFrame)
        
        # Check that it has some rows
        self.assertTrue(len(demographics) > 0)
        
        # Check that it has a geoid column
        self.assertTrue('geoid' in demographics.columns)
        
        # Check that it has demographic columns
        self.assertTrue('total_population' in demographics.columns)
    
    def test_merge_census_data(self):
        """Test merging census boundaries with demographic data."""
        # Skip if no API key available
        if not self.fetcher.api_key:
            self.skipTest("No Census API key available")
        
        # Get boundaries and demographics
        boundaries = self.fetcher.get_census_boundaries(self.test_state, self.test_county)
        demographics = self.fetcher.get_demographic_data(state=self.test_state, county=self.test_county)
        
        # Merge them
        merged = self.fetcher.merge_census_data(boundaries, demographics)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(merged, gpd.GeoDataFrame)
        
        # Check that it has geometries
        self.assertTrue('geometry' in merged.columns)
        
        # Check that it has demographic columns
        self.assertTrue('total_population' in merged.columns)
        
        # Check that the number of rows matches the boundaries
        self.assertEqual(len(merged), len(boundaries))
    
    def test_validate_census_data(self):
        """Test census data validation."""
        # Create a test dataframe with some issues
        test_data = pd.DataFrame({
            'geoid': ['53033001100', '53033001200', '53033001300'],
            'total_population': [1000, 0, 2000],
            'median_household_income': [50000, 1000000, 60000]
        })
        
        # Validate the data
        is_valid, issues = self.fetcher.validate_census_data(test_data)
        
        # Check that validation found issues
        self.assertFalse(is_valid)
        self.assertTrue('zero_population_tracts' in issues)
        self.assertTrue('unreasonable_income' in issues)
    
    def _compare_geodataframes(self, gdf1, gdf2, tolerance=1e-6):
        """Compare two GeoDataFrames with tolerance for floating point differences."""
        # Reset indices to ignore index differences
        gdf1 = gdf1.reset_index(drop=True)
        gdf2 = gdf2.reset_index(drop=True)
        
        # Check same shape
        if gdf1.shape != gdf2.shape:
            return False
        
        # Check same columns (ignoring order)
        if set(gdf1.columns) != set(gdf2.columns):
            return False
        
        # Check same CRS
        if gdf1.crs != gdf2.crs:
            return False
        
        # Check data columns (excluding geometry)
        data_cols = [col for col in gdf1.columns if col != 'geometry']
        for col in data_cols:
            if col in gdf1.columns and col in gdf2.columns:
                # Handle numeric columns with tolerance
                if gdf1[col].dtype in ['float64', 'float32'] or gdf2[col].dtype in ['float64', 'float32']:
                    if not np.allclose(gdf1[col].fillna(0), gdf2[col].fillna(0), rtol=tolerance, equal_nan=True):
                        return False
                else:
                    # For non-numeric columns, check exact equality
                    # Use values comparison to avoid index issues
                    if not (gdf1[col].values == gdf2[col].values).all():
                        return False
        
        # Check geometries with tolerance
        for i in range(len(gdf1)):
            geom1 = gdf1.iloc[i].geometry
            geom2 = gdf2.iloc[i].geometry
            if geom1 is None or geom2 is None:
                if geom1 != geom2:
                    return False
            else:
                # Use shapely's equals_exact for geometry comparison
                if not geom1.equals_exact(geom2, tolerance=tolerance):
                    return False
        
        return True
    
    def test_cache_mechanism(self):
        """Test that the cache mechanism works."""
        # Skip if no API key available
        if not self.fetcher.api_key:
            self.skipTest("No Census API key available")
        
        # First call should fetch from API
        boundaries1 = self.fetcher.get_census_boundaries(self.test_state, self.test_county)
        
        # Second call should use cache
        boundaries2 = self.fetcher.get_census_boundaries(self.test_state, self.test_county)
        
        # Both should be equivalent
        self.assertTrue(self._compare_geodataframes(boundaries1, boundaries2))
        
        # Force refresh by setting use_cache=False
        boundaries3 = self.fetcher.get_census_boundaries(self.test_state, self.test_county, use_cache=False)
        
        # Should still be equivalent to the original
        self.assertTrue(self._compare_geodataframes(boundaries1, boundaries3))

if __name__ == '__main__':
    unittest.main()