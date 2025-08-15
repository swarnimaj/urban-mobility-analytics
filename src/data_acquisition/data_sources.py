# File: src/data_acquisition/data_sources.py
"""
Data source connections and access functions for the Urban Mobility Analytics project.

This module provides unified access to:
1. Census API for demographic data
2. GTFS feeds for transit data
3. OpenStreetMap (via OSMnx) for street networks and amenities
"""
import os
import pandas as pd
import geopandas as gpd
import requests
import zipfile
import io
import osmnx as ox
from census import Census
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define target cities with their state FIPS codes
TARGET_CITIES = {
    "Seattle": {"state_fips": "53", "county_fips": "033", "state": "WA"},
    "Portland": {"state_fips": "41", "county_fips": "051", "state": "OR"},
    "San Francisco": {"state_fips": "06", "county_fips": "075", "state": "CA"},
}

# Define GTFS feed URLs for target cities
GTFS_FEEDS = {
    "Seattle": "https://kingcounty.gov/depts/transportation/metro/travel-options/bus/app-center/developer-resources.aspx",
    "Portland": "https://developer.trimet.org/GTFS.shtml",
    "San Francisco": "https://www.sfmta.com/reports/gtfs-transit-data",
}

class CensusDataSource:
    """Class to handle Census API connections and data retrieval."""
    
    def __init__(self, api_key=None):
        """Initialize with Census API key."""
        self.api_key = api_key or CENSUS_API_KEY
        if not self.api_key:
            raise ValueError("Census API key not found. Please add it to your .env file.")
        self.client = Census(self.api_key)
    
    def get_population_data(self, state_fips, county_fips, year=2021):
        """Get population data for a county."""
        try:
            data = self.client.acs5.state_county(
                fields=('NAME', 'B01001_001E'),  # B01001_001E is total population
                state_fips=state_fips,
                county_fips=county_fips,
                year=year
            )
            df = pd.DataFrame(data)
            df = df.rename(columns={'B01001_001E': 'total_population', 'NAME': 'name'})
            return df
        except Exception as e:
            logger.error(f"Error retrieving Census data: {e}")
            raise
    
    def get_demographic_data(self, state_fips, county_fips, year=2021):
        """Get demographic data for a county."""
        try:
            # Define demographic variables to retrieve
            variables = [
                'B01001_001E',  # Total population
                'B01001_002E',  # Male population
                'B01001_026E',  # Female population
                'B01002_001E',  # Median age
                'B02001_002E',  # White population
                'B02001_003E',  # Black population
                'B02001_004E',  # American Indian population
                'B02001_005E',  # Asian population
                'B19013_001E',  # Median household income
                'B25077_001E',  # Median home value
                'B08301_001E',  # Total commuters
                'B08301_010E',  # Public transit commuters
                'B08301_019E',  # Walk commuters
                'B08301_021E',  # Bicycle commuters
            ]
            
            data = self.client.acs5.state_county(
                fields=['NAME'] + variables,
                state_fips=state_fips,
                county_fips=county_fips,
                year=year
            )
            
            df = pd.DataFrame(data)
            
            # Rename columns for clarity
            column_names = {
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
                'NAME': 'name'
            }
            
            df = df.rename(columns=column_names)
            
            # Calculate percentages
            df['pct_public_transit'] = (df['public_transit_commuters'] / df['total_commuters']) * 100
            df['pct_walk'] = (df['walk_commuters'] / df['total_commuters']) * 100
            df['pct_bicycle'] = (df['bicycle_commuters'] / df['total_commuters']) * 100
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving Census demographic data: {e}")
            raise
    
    def get_census_tracts(self, state_fips, county_fips, year=2021):
        """Get Census tract boundaries for a county."""
        try:
            # This requires the Census Geocoder API or tigris package
            # For now, we'll use tigris via geopandas
            import geopandas as gpd
            
            # Cache file path
            cache_file = RAW_DATA_DIR / f"census_tracts_{state_fips}_{county_fips}_{year}.geojson"
            
            # Check if we already have the data cached
            if cache_file.exists():
                logger.info(f"Loading cached Census tract data from {cache_file}")
                gdf = gpd.read_file(cache_file)
                return gdf
            
            logger.info(f"Fetching Census tract boundaries for state {state_fips}, county {county_fips}")
            
            # Use the Census Geocoder API to get tract boundaries
            # Note: In a real implementation, you might want to use the tigris package
            # This is a simplified example
            from urllib.request import urlretrieve
            
            # URL for Census tract shapefile
            url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_{county_fips}_tract.zip"
            
            # Download and extract shapefile
            temp_zip = RAW_DATA_DIR / f"temp_tracts_{state_fips}_{county_fips}.zip"
            urlretrieve(url, temp_zip)
            
            # Read the shapefile
            gdf = gpd.read_file(f"zip://{temp_zip}")
            
            # Save to GeoJSON for caching
            gdf.to_file(cache_file, driver="GeoJSON")
            
            # Clean up temporary zip file
            os.remove(temp_zip)
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error retrieving Census tract boundaries: {e}")
            raise

class GTFSDataSource:
    """Class to handle GTFS feed access and processing."""
    
    def __init__(self):
        """Initialize GTFS data source."""
        pass
    
    def get_gtfs_feed_info(self, city):
        """Get information about GTFS feed for a city."""
        if city not in GTFS_FEEDS:
            raise ValueError(f"No GTFS feed URL defined for {city}")
        
        feed_url = GTFS_FEEDS[city]
        logger.info(f"GTFS feed for {city}: {feed_url}")
        
        # For now, just return the URL since most agencies require manual download
        # In a real implementation, you might automate this if direct download links are available
        return {
            "city": city,
            "feed_url": feed_url,
            "note": "Please visit this URL to download the GTFS feed manually."
        }
    
    def download_gtfs_feed(self, direct_url, city):
        """
        Download a GTFS feed from a direct download URL.
        
        Note: This only works if the agency provides a direct download link.
        Many agencies require registration or have download pages instead.
        """
        try:
            logger.info(f"Downloading GTFS feed for {city} from {direct_url}")
            
            # Create directory for GTFS data
            gtfs_dir = RAW_DATA_DIR / "gtfs" / city.lower().replace(" ", "_")
            gtfs_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the zip file
            response = requests.get(direct_url)
            response.raise_for_status()
            
            # Save the zip file
            zip_path = gtfs_dir / "gtfs.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(gtfs_dir)
            
            logger.info(f"GTFS feed downloaded and extracted to {gtfs_dir}")
            return gtfs_dir
            
        except Exception as e:
            logger.error(f"Error downloading GTFS feed: {e}")
            raise

class OSMDataSource:
    """Class to handle OpenStreetMap data access via OSMnx."""
    
    def __init__(self):
        """Initialize OSM data source."""
        # Configure OSMnx - removed deprecated config call
        # OSMnx no longer uses ox.config() in newer versions
    
    def get_street_network(self, place_name, network_type='drive'):
        """
        Get street network for a place.
        
        Args:
            place_name: Name of the place (e.g., "Seattle, Washington")
            network_type: Type of network ('drive', 'walk', 'bike', 'all')
            
        Returns:
            NetworkX graph of street network
        """
        try:
            logger.info(f"Fetching {network_type} street network for {place_name}")
            
            # Get the street network
            graph = ox.graph_from_place(place_name, network_type=network_type)
            
            logger.info(f"Retrieved street network with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
            
        except Exception as e:
            logger.error(f"Error retrieving OSM street network: {e}")
            raise
    
    def get_amenities(self, place_name, amenity_types=None):
        """
        Get amenities for a place.
        
        Args:
            place_name: Name of the place (e.g., "Seattle, Washington")
            amenity_types: List of amenity types to retrieve (e.g., ['school', 'hospital'])
            
        Returns:
            GeoDataFrame of amenities
        """
        try:
            if amenity_types is None:
                amenity_types = ['school', 'hospital', 'pharmacy', 'grocery', 'supermarket', 'bus_stop']
            
            logger.info(f"Fetching amenities for {place_name}: {amenity_types}")
            
            # Get the place boundary
            gdf_place = ox.geocode_to_gdf(place_name)
            
            # Get the amenities using OSMnx's geometries_from_polygon function
            # First, get the polygon of the place
            polygon = gdf_place.iloc[0]['geometry']
            
            # Create tags dictionary for amenity types
            tags = {'amenity': amenity_types}
            
            # Get the amenities within the polygon
            gdf = ox.features_from_polygon(polygon, tags)
            
            if gdf.empty:
                logger.warning(f"No amenities found for {place_name} with types {amenity_types}")
                return gdf
            
            logger.info(f"Retrieved {len(gdf)} amenities")
            return gdf
            
        except Exception as e:
            logger.error(f"Error retrieving OSM amenities: {e}")
            raise
    
    def get_accessibility_features(self, place_name):
        """
        Get accessibility-related features for a place.
        
        Args:
            place_name: Name of the place (e.g., "Seattle, Washington")
            
        Returns:
            GeoDataFrame of accessibility features
        """
        try:
            logger.info(f"Fetching accessibility features for {place_name}")
            
            # Get the place boundary
            gdf_place = ox.geocode_to_gdf(place_name)
            
            # Get the polygon of the place
            polygon = gdf_place.iloc[0]['geometry']
            
            # Define tags for accessibility features
            tags = {
                'wheelchair': ['yes', 'limited', 'designated'],
                'tactile_paving': ['yes'],
                'kerb': ['lowered', 'flush'],
                'highway': ['crossing', 'footway']
            }
            
            # Get the features within the polygon
            gdf = ox.features_from_polygon(polygon, tags)
            
            if gdf.empty:
                logger.warning(f"No accessibility features found for {place_name}")
                return gdf
            
            logger.info(f"Retrieved {len(gdf)} accessibility features")
            return gdf
            
        except Exception as e:
            logger.error(f"Error retrieving OSM accessibility features: {e}")
            raise

def test_all_data_sources(city="Seattle"):
    """Test all data sources for a city."""
    try:
        city_info = TARGET_CITIES.get(city)
        if not city_info:
            raise ValueError(f"City {city} not found in TARGET_CITIES")
        
        print(f"\n===== Testing data sources for {city} =====\n")
        
        # Test Census API
        print("\n----- Testing Census API -----")
        census = CensusDataSource()
        pop_data = census.get_population_data(city_info['state_fips'], city_info['county_fips'])
        print(f"Population data: {pop_data}")
        
        # Test GTFS feed info
        print("\n----- Testing GTFS Feed Info -----")
        gtfs = GTFSDataSource()
        feed_info = gtfs.get_gtfs_feed_info(city)
        print(f"GTFS feed info: {feed_info}")
        
        # Test OSM data (small area to avoid long download)
        print("\n----- Testing OSM Data -----")
        osm = OSMDataSource()
        
        # Use a smaller area for testing
        test_place = f"{city}, {city_info['state']}, USA"
        print(f"Testing with place: {test_place}")
        
        # Get a small sample of amenities
        amenities = osm.get_amenities(test_place, ['pharmacy'])
        if not amenities.empty:
            print(f"Successfully retrieved {len(amenities)} pharmacies")
            print(amenities.head(2))
        else:
            print("No pharmacies found in the area")
        
        print("\n===== All data source tests completed =====\n")
        return True
        
    except Exception as e:
        print(f"Error testing data sources: {e}")
        return False

if __name__ == "__main__":
    # Test all data sources for Seattle
    test_all_data_sources("Seattle")