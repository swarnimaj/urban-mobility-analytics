# File: src/data_acquisition/fetch_sample_data.py
"""
Script to fetch sample data from each data source.
"""
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from data_sources import CensusDataSource, GTFSDataSource, OSMDataSource, TARGET_CITIES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE_DATA_DIR = PROJECT_DIR / "data" / "interim" / "samples"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_census_sample(city="Seattle"):
    """Fetch sample Census data for a city."""
    logger.info(f"Fetching sample Census data for {city}")
    
    city_info = TARGET_CITIES.get(city)
    if not city_info:
        logger.error(f"City {city} not found in TARGET_CITIES")
        return None
    
    try:
        # Initialize Census data source
        census = CensusDataSource()
        
        # Get demographic data
        demo_data = census.get_demographic_data(
            city_info['state_fips'], 
            city_info['county_fips']
        )
        
        # Save to CSV
        output_file = SAMPLE_DATA_DIR / f"{city.lower()}_census_sample.csv"
        demo_data.to_csv(output_file, index=False)
        logger.info(f"Saved Census sample to {output_file}")
        
        return demo_data
    
    except Exception as e:
        logger.error(f"Error fetching Census sample: {e}")
        return None

def fetch_osm_sample(city="Seattle", amenity_type="pharmacy"):
    """Fetch sample OSM data for a city."""
    logger.info(f"Fetching sample OSM data for {city}")
    
    city_info = TARGET_CITIES.get(city)
    if not city_info:
        logger.error(f"City {city} not found in TARGET_CITIES")
        return None
    
    try:
        # Initialize OSM data source
        osm = OSMDataSource()
        
        # Get amenities
        place_name = f"{city}, {city_info['state']}, USA"
        amenities = osm.get_amenities(place_name, [amenity_type])
        
        if amenities.empty:
            logger.warning(f"No {amenity_type} amenities found for {city}")
            return None
        
        # Save to GeoJSON
        output_file = SAMPLE_DATA_DIR / f"{city.lower()}_{amenity_type}_osm_sample.geojson"
        amenities.to_file(output_file, driver="GeoJSON")
        logger.info(f"Saved OSM sample to {output_file}")
        
        return amenities
    
    except Exception as e:
        logger.error(f"Error fetching OSM sample: {e}")
        return None

def main():
    """Fetch sample data from all sources."""
    logger.info("Fetching sample data from all sources")
    
    # Fetch Census sample for Seattle
    census_sample = fetch_census_sample("Seattle")
    if census_sample is not None:
        logger.info(f"Census sample shape: {census_sample.shape}")
    
    # Fetch OSM sample for Seattle
    osm_sample = fetch_osm_sample("Seattle", "pharmacy")
    if osm_sample is not None:
        logger.info(f"OSM sample shape: {osm_sample.shape}")
    
    # Note: GTFS feeds typically require manual download
    logger.info("GTFS feeds need to be downloaded manually from transit agency websites")
    
    logger.info("Sample data fetching complete")

if __name__ == "__main__":
    main()