# File: src/data_acquisition/fetch_gtfs_data.py
"""
Example script to fetch and process GTFS data for a target city.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from gtfs_processor import GTFSProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "gtfs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_process_gtfs_data(agency_name="king_county_metro", agency_url=None):
    """
    Fetch and process GTFS data for a transit agency.
    
    Args:
        agency_name: Name of the transit agency
        agency_url: URL to the GTFS feed (optional)
        
    Returns:
        Dictionary containing stops GeoDataFrame and service frequency DataFrame
    """
    logger.info(f"Fetching GTFS data for {agency_name}")
    
    try:
        # Initialize the GTFS processor
        processor = GTFSProcessor(agency_name=agency_name)
        
        # Download the feed
        processor.download_gtfs_feed(agency_url=agency_url)
        
        # Load the feed
        feed = processor.load_gtfs_feed()
        
        # Extract stops
        stops = processor.extract_stops()
        
        # Convert to GeoDataFrame
        stops_gdf = processor.stops_to_geodataframe(stops)
        
        # Calculate service frequency
        frequency = processor.calculate_service_frequency()
        
        # Validate accessibility data
        is_valid, issues = processor.validate_accessibility_data(stops)
        if not is_valid:
            logger.warning(f"Accessibility data validation found issues: {issues}")
        
        # Save processed data
        stops_file = processor.save_processed_data(stops_gdf, "stops", file_format="geojson")
        frequency_file = processor.save_processed_data(frequency, "frequency", file_format="csv")
        
        # Create a simple visualization
        create_visualization(stops_gdf, frequency, agency_name)
        
        return {
            'stops': stops_gdf,
            'frequency': frequency,
            'feed': feed
        }
        
    except Exception as e:
        logger.error(f"Error processing GTFS data: {e}")
        raise

def process_multiple_agencies_for_city(city_name="Seattle"):
    """
    Process GTFS feeds from multiple agencies for a city.
    
    Args:
        city_name: Name of the city
        
    Returns:
        Dictionary containing merged stops GeoDataFrame and service frequency DataFrame
    """
    logger.info(f"Processing multiple agencies for {city_name}")
    
    try:
        # Define agencies for the city
        if city_name.lower() == "seattle":
            agency_urls = {
                "king_county_metro": "https://kingcounty.gov/~/media/depts/metro/data/gtfs/current-gtfs-zip",
                "sound_transit": "https://www.soundtransit.org/help-contacts/business-information/open-transit-data-otd/downloads"
            }
        else:
            raise ValueError(f"No agency information defined for {city_name}")
        
        # Create a processor
        processor = GTFSProcessor()
        
        # Process multiple agencies
        result = processor.process_multiple_agencies(agency_urls, city_name)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing multiple agencies: {e}")
        raise

def create_visualization(stops_gdf, frequency, agency_name):
    """Create a simple visualization of the GTFS data."""
    try:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot stops
        stops_gdf.plot(ax=axes[0], markersize=5, color='blue')
        axes[0].set_title(f"Transit Stops for {agency_name}")
        
        # Plot wheelchair accessible stops if data is available
        if 'wheelchair_accessible' in stops_gdf.columns:
            # Color by accessibility
            color_map = {'yes': 'green', 'no': 'red', 'unknown': 'gray'}
            stops_gdf.plot(ax=axes[1], markersize=5, column='wheelchair_accessible', 
                          categorical=True, legend=True, cmap='viridis')
            axes[1].set_title(f"Wheelchair Accessibility for {agency_name}")
        else:
            # Plot service frequency if available
            if not frequency.empty and 'stop_id' in frequency.columns:
                # Merge frequency with stops
                freq_gdf = stops_gdf.merge(frequency, on='stop_id', how='left')
                freq_gdf.plot(ax=axes[1], markersize=5, column='trips_per_hour', 
                             legend=True, cmap='viridis')
                axes[1].set_title(f"Service Frequency for {agency_name}")
        
        # Save the figure
        output_dir = OUTPUT_DIR / agency_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{agency_name}_visualization.png"
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")

if __name__ == "__main__":
    # Fetch and process GTFS data for King County Metro
    result = fetch_and_process_gtfs_data("king_county_metro")
    
    # Alternatively, process multiple agencies for Seattle
    # result = process_multiple_agencies_for_city("Seattle")