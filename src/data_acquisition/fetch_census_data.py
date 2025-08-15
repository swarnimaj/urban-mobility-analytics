# File: src/data_acquisition/census_fetcher.py
"""
Census data acquisition module for the Urban Mobility Analytics project.

This module provides functions to:
1. Fetch census tract boundaries
2. Fetch demographic data
3. Merge boundaries with demographic data
4. Validate and cache census data
"""

import os
import time
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from census import Census
from dotenv import load_dotenv
import logging
import json
import requests
from urllib.request import urlretrieve
import zipfile
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Census API key
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    logger.warning("Census API key not found. Please add it to your .env file.")

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "census"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed" / "census"
CACHE_DIR = PROJECT_DIR / "data" / "interim" / "census_cache"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Define default demographic variables to fetch
DEFAULT_VARIABLES = {
    'B01001_001E': 'total_population',
    'B01001_002E': 'male_population',
    'B01001_026E': 'female_population',
    'B01002_001E': 'median_age',
    'B02001_002E': 'white_population',
    'B02001_003E': 'black_population',
    'B02001_004E': 'american_indian_population',
    'B02001_005E': 'asian_population',
    'B19013_001E': 'median_household_income',
    'B25077_001E': 'median_home_value',
    'B08301_001E': 'total_commuters',
    'B08301_010E': 'public_transit_commuters',
    'B08301_019E': 'walk_commuters',
    'B08301_021E': 'bicycle_commuters',
    'B18101_001E': 'total_disability_status_determined',
    'B18101_004E': 'male_with_disability_under_5',
    'B18101_007E': 'male_with_disability_5_to_17',
    'B18101_010E': 'male_with_disability_18_to_34',
    'B18101_013E': 'male_with_disability_35_to_64',
    'B18101_016E': 'male_with_disability_65_to_74',
    'B18101_019E': 'male_with_disability_75_plus',
    'B18101_023E': 'female_with_disability_under_5',
    'B18101_026E': 'female_with_disability_5_to_17',
    'B18101_029E': 'female_with_disability_18_to_34',
    'B18101_032E': 'female_with_disability_35_to_64',
    'B18101_035E': 'female_with_disability_65_to_74',
    'B18101_038E': 'female_with_disability_75_plus',
}

# Define cache expiration time (in days)
CACHE_EXPIRATION_DAYS = 30

class CensusFetcher:
    """Class for fetching and processing Census data."""
    
    def __init__(self, api_key=None, year=2021):
        """
        Initialize the Census fetcher.
        
        Args:
            api_key: Census API key (defaults to environment variable)
            year: Census data year (defaults to 2021)
        """
        self.api_key = api_key or CENSUS_API_KEY
        if not self.api_key:
            raise ValueError("Census API key not found. Please add it to your .env file.")
        
        self.census = Census(self.api_key)
        self.year = year
        logger.info(f"Initialized Census fetcher for year {self.year}")
    
    def get_census_boundaries(self, state, county, use_cache=True):
        """
        Get census tract boundaries for a county.
        
        Args:
            state: State FIPS code or name
            county: County FIPS code or name
            use_cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame of census tract boundaries
        """
        # Convert state and county to FIPS codes if they are names
        state_fips = self._get_state_fips(state)
        county_fips = self._get_county_fips(state_fips, county)
        
        # Generate cache file path
        cache_file = CACHE_DIR / f"boundaries_{state_fips}_{county_fips}_{self.year}.geojson"
        
        # Check if cache file exists and is not expired
        if use_cache and self._is_cache_valid(cache_file):
            logger.info(f"Loading cached census boundaries from {cache_file}")
            return gpd.read_file(cache_file)
        
        logger.info(f"Fetching census tract boundaries for state {state_fips}, county {county_fips}")
        
        try:
            # Path to the manually downloaded shapefile directory
            tiger_dir = RAW_DATA_DIR / "tiger" / f"tl_{self.year}_{state_fips}_tract"
            shp_file = tiger_dir / f"tl_{self.year}_{state_fips}_tract.shp"
            
            # Check if the file exists
            if not shp_file.exists():
                raise FileNotFoundError(
                    f"Census tract shapefile not found at {shp_file}. "
                )
            
            # Read the shapefile directly
            gdf = gpd.read_file(shp_file)
            
            # Filter for the specific county
            gdf = gdf[gdf['COUNTYFP'] == county_fips]
            
            if len(gdf) == 0:
                raise ValueError(f"No census tracts found for county {county_fips} in state {state_fips}")
            
            # Rename GEOID column for consistency
            if 'GEOID' in gdf.columns:
                gdf = gdf.rename(columns={'GEOID': 'geoid'})
            elif 'GEOID10' in gdf.columns:
                gdf = gdf.rename(columns={'GEOID10': 'geoid'})
            
            # Ensure the CRS is WGS84
            gdf = gdf.to_crs("EPSG:4326")
            
            # Save to GeoJSON for caching
            gdf.to_file(cache_file, driver="GeoJSON")
            
            logger.info(f"Retrieved {len(gdf)} census tracts")
            return gdf
            
        except Exception as e:
            logger.error(f"Error retrieving Census tract boundaries: {e}")
            raise
    
    def get_demographic_data(self, variables=None, state=None, county=None, use_cache=True):
        """
        Get demographic data for a county at the census tract level.
        
        Args:
            variables: List of Census variables to fetch (defaults to DEFAULT_VARIABLES)
            state: State FIPS code or name
            county: County FIPS code or name
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame of demographic data
        """
        # Use default variables if none provided
        if variables is None:
            variables = list(DEFAULT_VARIABLES.keys())
        
        # Convert state and county to FIPS codes if they are names
        state_fips = self._get_state_fips(state)
        county_fips = self._get_county_fips(state_fips, county)
        
        # Generate cache file path
        var_hash = hash(tuple(sorted(variables)))
        cache_file = CACHE_DIR / f"demographics_{state_fips}_{county_fips}_{var_hash}_{self.year}.csv"
        
        # Check if cache file exists and is not expired
        if use_cache and self._is_cache_valid(cache_file):
            logger.info(f"Loading cached demographic data from {cache_file}")
            return pd.read_csv(cache_file)
        
        logger.info(f"Fetching demographic data for state {state_fips}, county {county_fips}")
        
        try:
            # Fetch data from Census API
            data = self.census.acs5.state_county_tract(
                fields=['NAME'] + variables,
                state_fips=state_fips,
                county_fips=county_fips,
                tract='*',
                year=self.year
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create geoid column for joining with boundaries
            df['geoid'] = df['state'] + df['county'] + df['tract']
            
            # Rename columns if variable mapping is provided
            if variables == list(DEFAULT_VARIABLES.keys()):
                rename_dict = DEFAULT_VARIABLES.copy()
                rename_dict['NAME'] = 'name'
                df = df.rename(columns=rename_dict)
            
            # Save to CSV for caching
            df.to_csv(cache_file, index=False)
            
            logger.info(f"Retrieved demographic data for {len(df)} census tracts")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving demographic data: {e}")
            # Try to use backup method if API fails
            return self._get_demographic_data_backup(state_fips, county_fips, variables)
    
    def merge_census_data(self, boundaries=None, demographics=None, state=None, county=None):
        """
        Merge census boundaries with demographic data.
        
        Args:
            boundaries: GeoDataFrame of census tract boundaries (optional)
            demographics: DataFrame of demographic data (optional)
            state: State FIPS code or name (required if boundaries or demographics not provided)
            county: County FIPS code or name (required if boundaries or demographics not provided)
            
        Returns:
            GeoDataFrame with merged census data
        """
        # Fetch boundaries and demographics if not provided
        if boundaries is None:
            if state is None or county is None:
                raise ValueError("Must provide either boundaries or state and county")
            boundaries = self.get_census_boundaries(state, county)
        
        if demographics is None:
            if state is None or county is None:
                raise ValueError("Must provide either demographics or state and county")
            demographics = self.get_demographic_data(state=state, county=county)
        
        logger.info("Merging census boundaries with demographic data")
        
        try:
            # Ensure geoid columns are the same type for merging
            boundaries['geoid'] = boundaries['geoid'].astype(str)
            demographics['geoid'] = demographics['geoid'].astype(str)
            
            # Merge boundaries with demographics on geoid
            merged = boundaries.merge(demographics, on='geoid', how='left')
            
            # Calculate derived metrics
            if 'total_commuters' in merged.columns and merged['total_commuters'].sum() > 0:
                # Calculate commuting percentages
                if 'public_transit_commuters' in merged.columns:
                    merged['pct_public_transit'] = (merged['public_transit_commuters'] / merged['total_commuters']) * 100
                
                if 'walk_commuters' in merged.columns:
                    merged['pct_walk'] = (merged['walk_commuters'] / merged['total_commuters']) * 100
                
                if 'bicycle_commuters' in merged.columns:
                    merged['pct_bicycle'] = (merged['bicycle_commuters'] / merged['total_commuters']) * 100
            
            # Calculate disability percentage
            disability_cols = [col for col in merged.columns if 'with_disability' in col]
            if disability_cols and 'total_disability_status_determined' in merged.columns:
                merged['total_with_disability'] = merged[disability_cols].sum(axis=1)
                merged['pct_with_disability'] = (merged['total_with_disability'] / merged['total_disability_status_determined']) * 100
            
            logger.info(f"Successfully merged data for {len(merged)} census tracts")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging census data: {e}")
            raise
    
    def validate_census_data(self, data):
        """
        Validate census data for common issues.
        
        Args:
            data: DataFrame or GeoDataFrame of census data
            
        Returns:
            Tuple of (is_valid, issues_dict)
        """
        issues = {}
        
        # Check for missing geometries if it's a GeoDataFrame
        if isinstance(data, gpd.GeoDataFrame):
            missing_geoms = int(data.geometry.isna().sum())
            if missing_geoms > 0:
                issues['missing_geometries'] = missing_geoms
        
        # Check for missing demographic values
        for col in data.columns:
            if col not in ['geometry', 'name', 'geoid', 'state', 'county', 'tract']:
                missing = int(data[col].isna().sum())
                if missing > 0:
                    issues[f'missing_{col}'] = missing
        
        # Check for unreasonable values
        if 'median_household_income' in data.columns:
            unreasonable = int(data[data['median_household_income'] > 500000].shape[0])
            if unreasonable > 0:
                issues['unreasonable_income'] = unreasonable
        
        if 'total_population' in data.columns:
            zero_pop = int(data[data['total_population'] == 0].shape[0])
            if zero_pop > 0:
                issues['zero_population_tracts'] = zero_pop
        
        # Overall validation result
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Census data validation passed")
        else:
            logger.warning(f"Census data validation found issues: {issues}")
        
        return is_valid, issues
    
    def save_processed_data(self, data, state, county):
        """
        Save processed census data to the processed data directory.
        
        Args:
            data: GeoDataFrame of processed census data
            state: State FIPS code or name
            county: County FIPS code or name
            
        Returns:
            Path to saved file
        """
        # Convert state and county to FIPS codes if they are names
        state_fips = self._get_state_fips(state)
        county_fips = self._get_county_fips(state_fips, county)
        
        # Generate file path
        file_path = PROCESSED_DATA_DIR / f"census_{state_fips}_{county_fips}_{self.year}.geojson"
        
        # Save data
        data.to_file(file_path, driver="GeoJSON")
        logger.info(f"Saved processed census data to {file_path}")
        
        return file_path
    
    def _get_state_fips(self, state):
        """Convert state name to FIPS code if needed."""
        if state is None:
            raise ValueError("State must be provided")
        
        # If state is already a FIPS code (2 digits)
        if isinstance(state, str) and state.isdigit() and len(state) == 2:
            return state
        
        # State name to FIPS mapping
        state_fips = {
            'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
            'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17',
            'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
            'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31',
            'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
            'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46',
            'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
            'WI': '55', 'WY': '56', 'AS': '60', 'GU': '66', 'MP': '69', 'PR': '72', 'VI': '78'
        }
        
        # Convert 2-letter state code to FIPS
        if isinstance(state, str) and len(state) == 2 and state.upper() in state_fips:
            return state_fips[state.upper()]
        
        # Handle full state names
        state_names = {
            'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05', 'california': '06',
            'colorado': '08', 'connecticut': '09', 'delaware': '10', 'district of columbia': '11',
            'florida': '12', 'georgia': '13', 'hawaii': '15', 'idaho': '16', 'illinois': '17',
            'indiana': '18', 'iowa': '19', 'kansas': '20', 'kentucky': '21', 'louisiana': '22',
            'maine': '23', 'maryland': '24', 'massachusetts': '25', 'michigan': '26', 'minnesota': '27',
            'mississippi': '28', 'missouri': '29', 'montana': '30', 'nebraska': '31', 'nevada': '32',
            'new hampshire': '33', 'new jersey': '34', 'new mexico': '35', 'new york': '36',
            'north carolina': '37', 'north dakota': '38', 'ohio': '39', 'oklahoma': '40', 'oregon': '41',
            'pennsylvania': '42', 'rhode island': '44', 'south carolina': '45', 'south dakota': '46',
            'tennessee': '47', 'texas': '48', 'utah': '49', 'vermont': '50', 'virginia': '51',
            'washington': '53', 'west virginia': '54', 'wisconsin': '55', 'wyoming': '56'
        }
        
        if isinstance(state, str) and state.lower() in state_names:
            return state_names[state.lower()]
        
        raise ValueError(f"Invalid state: {state}")
    
    def _get_county_fips(self, state_fips, county):
        """Convert county name to FIPS code if needed."""
        if county is None:
            raise ValueError("County must be provided")
        
        # If county is already a FIPS code (3 digits)
        if isinstance(county, str) and county.isdigit() and len(county) == 3:
            return county
        
        # If county is a name, we need to look it up
        # This is a simplified version - in a real implementation, you might want to use a more comprehensive lookup
        try:
            # Get all counties in the state
            counties = self.census.acs5.state_county(
                fields=['NAME'],
                state_fips=state_fips,
                county_fips='*',
                year=self.year
            )
            
            # Find the matching county
            for c in counties:
                county_name = c['NAME'].split(',')[0].lower()
                if county.lower() in county_name or county_name in county.lower():
                    return c['county']
            
            raise ValueError(f"County not found: {county} in state {state_fips}")
            
        except Exception as e:
            logger.error(f"Error looking up county FIPS: {e}")
            raise ValueError(f"Could not determine FIPS code for county: {county}")
    
    def _is_cache_valid(self, cache_file):
        """Check if a cache file exists and is not expired."""
        if not cache_file.exists():
            return False
        
        # Check if file is older than expiration time
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        expiration_time = datetime.now() - timedelta(days=CACHE_EXPIRATION_DAYS)
        
        return file_time > expiration_time
    
    def _get_demographic_data_backup(self, state_fips, county_fips, variables):
        """Backup method to get demographic data if the API fails."""
        logger.warning("Using backup method to get demographic data")
        
        try:
            # Try to use the Census Data API directly with requests
            url = f"https://api.census.gov/data/{self.year}/acs/acs5"
            
            # Prepare variables string
            vars_str = ",".join(['NAME'] + variables)
            
            # Make the request
            params = {
                'get': vars_str,
                'for': f'tract:*',
                'in': f'state:{state_fips} county:{county_fips}',
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            headers = data[0]
            rows = data[1:]
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Create geoid column
            df['geoid'] = df['state'] + df['county'] + df['tract']
            
            # Rename columns if variable mapping is provided
            if set(variables) == set(DEFAULT_VARIABLES.keys()):
                rename_dict = DEFAULT_VARIABLES.copy()
                rename_dict['NAME'] = 'name'
                df = df.rename(columns=rename_dict)
            
            # Convert numeric columns to float
            for var in variables:
                if var in df.columns:
                    df[var] = pd.to_numeric(df[var], errors='coerce')
            
            logger.info(f"Retrieved demographic data for {len(df)} census tracts using backup method")
            return df
            
        except Exception as e:
            logger.error(f"Backup method also failed: {e}")
            raise