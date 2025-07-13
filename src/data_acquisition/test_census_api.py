# File: src/data_acquisition/test_census_api.py
"""
Test script to verify Census API access.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from census import Census
import json

# Load environment variables from .env file
load_dotenv()

# Get Census API key from environment variables
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

if not CENSUS_API_KEY:
    raise ValueError("Census API key not found. Please add it to your .env file.")

def test_census_connection():
    """Test connection to Census API and basic query."""
    print("Testing Census API connection...")
    
    try:
        # Initialize Census API client
        c = Census(CENSUS_API_KEY)
        
        # Test a simple query - get population data for King County, WA (FIPS code 53033)
        # Using American Community Survey (ACS) 5-year data from 2021
        data = c.acs5.state_county(
            fields=('NAME', 'B01001_001E'),  # B01001_001E is total population
            state_fips='53',                 # Washington state FIPS code
            county_fips='033',               # King County FIPS code
            year=2021
        )
        
        # Convert to DataFrame for easier viewing
        df = pd.DataFrame(data)
        
        # Rename columns for clarity
        df = df.rename(columns={'B01001_001E': 'total_population', 'NAME': 'name'})
        
        print("\nSuccessfully retrieved data from Census API:")
        print(df)
        print("\nCensus API connection test: SUCCESS")
        return True
        
    except Exception as e:
        print(f"\nCensus API connection test: FAILED")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_census_connection()