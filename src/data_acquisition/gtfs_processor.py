# File: src/data_acquisition/gtfs_processor.py
"""
GTFS transit data processing module for the Urban Mobility Analytics project.

This module provides functions to:
1. Download GTFS feeds from transit agencies
2. Extract transit stops with accessibility information
3. Convert stops to GeoDataFrame for spatial analysis
4. Calculate service frequency and other metrics
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
import logging
from datetime import datetime, timedelta
import tempfile
import shutil
import warnings
from shapely.geometry import Point
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "gtfs"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed" / "gtfs"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define known GTFS feed URLs for major transit agencies
GTFS_FEED_URLS = {
    "king_county_metro": "https://kingcounty.gov/~/media/depts/metro/data/gtfs/current-gtfs-zip",
    "sound_transit": "https://www.soundtransit.org/GTFS-rail/40_gtfs.zip",
    "community_transit": "https://www.soundtransit.org/GTFS-CT/current.zip",
    "pierce_transit": "https://www.soundtransit.org/GTFS-PT/gtfs.zip",
    "portland_trimet": "https://developer.trimet.org/schedule/gtfs.zip",
    "sf_muni": "https://gtfs.sfmta.com/transitdata/google_transit.zip"
}

class GTFSProcessor:
    """Class for processing GTFS transit data."""
    
    def __init__(self, agency_name=None, gtfs_dir=None):
        """
        Initialize the GTFS processor.
        
        Args:
            agency_name: Name of the transit agency (used for file naming)
            gtfs_dir: Directory containing GTFS files (if already downloaded)
        """
        self.agency_name = agency_name
        self.gtfs_dir = gtfs_dir
        self.feed = None
        
        # Create a subdirectory for this agency if needed
        if agency_name and not gtfs_dir:
            self.gtfs_dir = RAW_DATA_DIR / agency_name
            self.gtfs_dir.mkdir(parents=True, exist_ok=True)
    
    def download_gtfs_feed(self, agency_url=None, output_dir=None, force_download=False):
        """
        Download a GTFS feed from a transit agency.
        
        Args:
            agency_url: URL to the GTFS zip file
            output_dir: Directory to save the downloaded feed
            force_download: Whether to download even if files exist
            
        Returns:
            Path to the directory containing the extracted GTFS files
        """
        # Use provided URL or look up from known agencies
        if not agency_url and self.agency_name in GTFS_FEED_URLS:
            agency_url = GTFS_FEED_URLS[self.agency_name]
        
        if not agency_url:
            raise ValueError("No agency URL provided and agency name not found in known feeds")
        
        # Use provided output directory or default
        output_dir = output_dir or self.gtfs_dir
        if not output_dir:
            if not self.agency_name:
                self.agency_name = "unknown_agency"
            output_dir = RAW_DATA_DIR / self.agency_name
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if files already exist
        stops_file = output_dir / "stops.txt"
        if not force_download and stops_file.exists():
            logger.info(f"GTFS files already exist in {output_dir}. Use force_download=True to re-download.")
            self.gtfs_dir = output_dir
            return output_dir
        
        logger.info(f"Downloading GTFS feed from {agency_url}")
        
        try:
            # Download the zip file
            response = requests.get(agency_url)
            response.raise_for_status()
            
            # Save the zip file
            zip_path = output_dir / "gtfs.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            logger.info(f"GTFS feed downloaded and extracted to {output_dir}")
            self.gtfs_dir = output_dir
            return output_dir
            
        except Exception as e:
            logger.error(f"Error downloading GTFS feed: {e}")
            raise
    
    def load_gtfs_feed(self, gtfs_dir=None):
        """
        Load a GTFS feed from a directory.
        
        Args:
            gtfs_dir: Directory containing GTFS files
            
        Returns:
            Dictionary of DataFrames for each GTFS file
        """
        # Use provided directory or default
        gtfs_dir = gtfs_dir or self.gtfs_dir
        if not gtfs_dir:
            raise ValueError("No GTFS directory provided")
        
        logger.info(f"Loading GTFS feed from {gtfs_dir}")
        
        try:
            # List of core GTFS files
            core_files = ['stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt', 'calendar.txt']
            
            # Dictionary to store DataFrames
            feed = {}
            
            # Load each file if it exists
            for file_name in os.listdir(gtfs_dir):
                if file_name.endswith('.txt'):
                    file_path = gtfs_dir / file_name
                    try:
                        # Read the file with low_memory=False to avoid dtype warnings
                        df = pd.read_csv(file_path, low_memory=False)
                        # Store in dictionary with file name as key (without .txt)
                        feed[file_name[:-4]] = df
                    except Exception as e:
                        logger.warning(f"Error reading {file_name}: {e}")
            
            # Check that all core files were loaded
            missing_files = [f for f in core_files if f[:-4] not in feed]
            if missing_files:
                logger.warning(f"Missing core GTFS files: {missing_files}")
            
            self.feed = feed
            return feed
            
        except Exception as e:
            logger.error(f"Error loading GTFS feed: {e}")
            raise
    
    def extract_stops(self, feed=None, accessibility=True):
        """
        Extract transit stops from a GTFS feed, optionally filtering for accessibility.
        
        Args:
            feed: Dictionary of GTFS DataFrames (if not already loaded)
            accessibility: Whether to include accessibility information
            
        Returns:
            DataFrame of transit stops
        """
        # Use provided feed or loaded feed
        feed = feed or self.feed
        if not feed:
            raise ValueError("No GTFS feed provided or loaded")
        
        if 'stops' not in feed:
            raise ValueError("GTFS feed does not contain stops.txt")
        
        logger.info("Extracting stops from GTFS feed")
        
        try:
            # Get stops DataFrame
            stops = feed['stops'].copy()
            
            # Check if wheelchair_boarding column exists
            if accessibility and 'wheelchair_boarding' in stops.columns:
                # Filter for accessible stops if requested
                logger.info("Including wheelchair accessibility information")
                
                # In GTFS, wheelchair_boarding values are:
                # 0 or empty = No information
                # 1 = Some vehicles at this stop can be boarded by a rider in a wheelchair
                # 2 = Wheelchair boarding is not possible at this stop
                
                # Convert to more readable values
                # Handle both integer and float values
                stops['wheelchair_accessible'] = stops['wheelchair_boarding'].astype(float).map({
                    0.0: 'unknown',
                    1.0: 'yes',
                    2.0: 'no'
                }).fillna('unknown')
                
            elif accessibility:
                logger.warning("Wheelchair accessibility information not available in this feed")
                stops['wheelchair_accessible'] = 'unknown'
            
            # Ensure required columns exist
            required_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
            missing_cols = [col for col in required_cols if col not in stops.columns]
            if missing_cols:
                raise ValueError(f"Stops data is missing required columns: {missing_cols}")
            
            logger.info(f"Extracted {len(stops)} stops from GTFS feed")
            return stops
            
        except Exception as e:
            logger.error(f"Error extracting stops: {e}")
            raise
    
    def stops_to_geodataframe(self, stops=None):
        """
        Convert stops DataFrame to GeoDataFrame.
        
        Args:
            stops: DataFrame of stops (if not already extracted)
            
        Returns:
            GeoDataFrame of stops with Point geometries
        """
        # Extract stops if not provided
        if stops is None:
            stops = self.extract_stops()
        
        logger.info("Converting stops to GeoDataFrame")
        
        try:
            # Check for required columns
            if 'stop_lat' not in stops.columns or 'stop_lon' not in stops.columns:
                raise ValueError("Stops data must contain stop_lat and stop_lon columns")
            
            # Create Point geometries
            geometry = [Point(lon, lat) for lon, lat in zip(stops['stop_lon'], stops['stop_lat'])]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs="EPSG:4326")
            
            # Ensure consistent data types for problematic columns
            # Convert zone_id to string to handle mixed types
            if 'zone_id' in gdf.columns:
                gdf['zone_id'] = gdf['zone_id'].astype(str)
            
            # Convert other potentially problematic columns
            problematic_cols = ['stop_url', 'parent_station', 'stop_desc']
            for col in problematic_cols:
                if col in gdf.columns:
                    gdf[col] = gdf[col].astype(str)
            
            logger.info(f"Created GeoDataFrame with {len(gdf)} stops")
            return gdf
            
        except Exception as e:
            logger.error(f"Error creating GeoDataFrame: {e}")
            raise
    
    def calculate_service_frequency(self, stop_id=None, date=None, feed=None):
        """
        Calculate service frequency for a stop or all stops.
        
        Args:
            stop_id: ID of the stop to calculate frequency for (None for all stops)
            date: Date to calculate frequency for (defaults to current date)
            feed: Dictionary of GTFS DataFrames (if not already loaded)
            
        Returns:
            DataFrame with service frequency information
        """
        # Use provided feed or loaded feed
        feed = feed or self.feed
        if not feed:
            raise ValueError("No GTFS feed provided or loaded")
        
        # Check for required files
        required_files = ['stops', 'stop_times', 'trips', 'calendar']
        missing_files = [f for f in required_files if f not in feed]
        if missing_files:
            raise ValueError(f"GTFS feed is missing required files: {missing_files}")
        
        # Use current date if none provided
        if date is None:
            date = datetime.now().date()
        
        logger.info(f"Calculating service frequency for date {date}")
        
        try:
            # Get day of week
            day_of_week = date.strftime("%A").lower()
            
            # Filter calendar for services active on the given date
            calendar = feed['calendar'].copy()
            
            # Convert start and end dates to datetime
            calendar['start_date'] = pd.to_datetime(calendar['start_date'], format='%Y%m%d')
            calendar['end_date'] = pd.to_datetime(calendar['end_date'], format='%Y%m%d')
            
            # Filter for services active on the given date and day of week
            date_filter = (calendar['start_date'] <= pd.Timestamp(date)) & (calendar['end_date'] >= pd.Timestamp(date))
            day_filter = calendar[day_of_week] == 1
            active_services = calendar.loc[date_filter & day_filter, 'service_id'].tolist()
            
            # Filter trips for active services
            trips = feed['trips'].copy()
            active_trips = trips[trips['service_id'].isin(active_services)]
            
            # Get stop times for active trips
            stop_times = feed['stop_times'].copy()
            active_stop_times = stop_times[stop_times['trip_id'].isin(active_trips['trip_id'])]
            
            # Filter for specific stop if provided
            if stop_id:
                active_stop_times = active_stop_times[active_stop_times['stop_id'] == stop_id]
            
            # Group by stop_id and count trips
            frequency = active_stop_times.groupby('stop_id').size().reset_index(name='trips_per_day')
            
            # Calculate actual service hours for each stop
            # Convert arrival_time to hours and find service span
            active_stop_times = active_stop_times.copy()  # Create a copy to avoid SettingWithCopyWarning
            active_stop_times['arrival_hour'] = pd.to_numeric(
                active_stop_times['arrival_time'].str.split(':').str[0], 
                errors='coerce'
            )
            
            # Calculate service hours for each stop
            service_hours = active_stop_times.groupby('stop_id')['arrival_hour'].agg([
                ('first_service', 'min'),
                ('last_service', 'max')
            ]).reset_index()
            
            # Calculate service span (with minimum 1 hour)
            service_hours['service_span_hours'] = (
                service_hours['last_service'] - service_hours['first_service'] + 1
            ).clip(lower=1)
            
            # Merge service hours with frequency data
            frequency = frequency.merge(service_hours[['stop_id', 'service_span_hours']], on='stop_id', how='left')
            
            # Calculate trips per hour using actual service hours
            frequency['trips_per_hour'] = frequency['trips_per_day'] / frequency['service_span_hours']
            
            # Calculate average headway in minutes (with safety check for zero trips)
            frequency['avg_headway_minutes'] = np.where(
                frequency['trips_per_hour'] > 0,
                60 / frequency['trips_per_hour'],
                1440  # 24 hours if no service
            )
            
            # Merge with stops to get stop names and locations
            stops = feed['stops'].copy()
            frequency = frequency.merge(stops[['stop_id', 'stop_name']], on='stop_id')
            
            logger.info(f"Calculated service frequency for {len(frequency)} stops")
            return frequency
            
        except Exception as e:
            logger.error(f"Error calculating service frequency: {e}")
            raise
    
    def validate_accessibility_data(self, stops=None):
        """
        Validate wheelchair accessibility data in stops.
        
        Args:
            stops: DataFrame of stops (if not already extracted)
            
        Returns:
            Tuple of (is_valid, issues_dict)
        """
        # Extract stops if not provided
        if stops is None:
            stops = self.extract_stops()
        
        logger.info("Validating wheelchair accessibility data")
        
        issues = {}
        
        try:
            # Check if wheelchair_boarding column exists
            if 'wheelchair_boarding' not in stops.columns:
                issues['missing_wheelchair_data'] = "wheelchair_boarding column not found"
                return False, issues
            
            # Count values for each accessibility category
            accessibility_counts = stops['wheelchair_boarding'].value_counts().to_dict()
            
            # Check for invalid values
            valid_values = [0, 1, 2]
            invalid_values = [val for val in accessibility_counts.keys() 
                             if val not in valid_values and not pd.isna(val)]
            
            if invalid_values:
                issues['invalid_wheelchair_values'] = invalid_values
            
            # Check for missing values
            missing_values = stops['wheelchair_boarding'].isna().sum()
            if missing_values > 0:
                issues['missing_wheelchair_values'] = missing_values
            
            # Check for stops with no accessibility information
            unknown_accessibility = stops[stops['wheelchair_boarding'] == 0].shape[0]
            if unknown_accessibility > 0:
                issues['unknown_accessibility'] = unknown_accessibility
            
            # Check for stops that are explicitly not wheelchair accessible
            not_accessible = stops[stops['wheelchair_boarding'] == 2].shape[0]
            if not_accessible > 0:
                issues['not_accessible_stops'] = not_accessible
            
            # Calculate percentage of stops with known accessibility info
            total_stops = len(stops)
            known_accessibility = stops[stops['wheelchair_boarding'].isin([1, 2])].shape[0]
            pct_known = (known_accessibility / total_stops) * 100 if total_stops > 0 else 0
            
            if pct_known < 50:
                issues['low_accessibility_coverage'] = f"Only {pct_known:.1f}% of stops have known accessibility information"
            
            # Overall validation result
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info("Wheelchair accessibility data validation passed")
            else:
                logger.warning(f"Wheelchair accessibility data validation found issues: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating accessibility data: {e}")
            issues['validation_error'] = str(e)
            return False, issues
        
    def merge_agency_feeds(self, agency_feeds):
        """
        Merge GTFS feeds from multiple agencies.
        
        Args:
            agency_feeds: Dictionary mapping agency names to their GTFS feed dictionaries
            
        Returns:
            Merged GTFS feed dictionary
        """
        if not agency_feeds:
            raise ValueError("No agency feeds provided")
        
        logger.info(f"Merging feeds from {len(agency_feeds)} agencies")
        
        try:
            # Initialize merged feed with empty DataFrames
            merged_feed = {}
            
            # List of files to merge
            files_to_merge = ['stops', 'routes', 'trips', 'stop_times', 'calendar']
            
            # Add agency column to each DataFrame and merge
            for agency_name, feed in agency_feeds.items():
                for file_name in files_to_merge:
                    if file_name in feed:
                        # Make a copy of the DataFrame
                        df = feed[file_name].copy()
                        
                        # Add agency column if it doesn't exist
                        if 'agency_name' not in df.columns:
                            df['agency_name'] = agency_name
                        
                        # Add to merged feed
                        if file_name not in merged_feed:
                            merged_feed[file_name] = df
                        else:
                            # Check for ID conflicts
                            if file_name == 'stops' and 'stop_id' in df.columns:
                                # Prefix stop_ids with agency name to avoid conflicts
                                df['original_stop_id'] = df['stop_id']
                                df['stop_id'] = agency_name + '_' + df['stop_id'].astype(str)
                            
                            if file_name == 'routes' and 'route_id' in df.columns:
                                # Prefix route_ids with agency name to avoid conflicts
                                df['original_route_id'] = df['route_id']
                                df['route_id'] = agency_name + '_' + df['route_id'].astype(str)
                            
                            if file_name == 'trips' and 'trip_id' in df.columns:
                                # Prefix trip_ids with agency name to avoid conflicts
                                df['original_trip_id'] = df['trip_id']
                                df['trip_id'] = agency_name + '_' + df['trip_id'].astype(str)
                                
                                # Also update route_ids if they were prefixed
                                if 'route_id' in df.columns:
                                    df['route_id'] = agency_name + '_' + df['route_id'].astype(str)
                            
                            if file_name == 'stop_times':
                                # Update stop_ids and trip_ids to match the prefixed versions
                                if 'stop_id' in df.columns:
                                    df['stop_id'] = agency_name + '_' + df['stop_id'].astype(str)
                                if 'trip_id' in df.columns:
                                    df['trip_id'] = agency_name + '_' + df['trip_id'].astype(str)
                            
                            # Append to merged DataFrame
                            merged_feed[file_name] = pd.concat([merged_feed[file_name], df], ignore_index=True)
            
            logger.info(f"Successfully merged feeds from {len(agency_feeds)} agencies")
            return merged_feed
            
        except Exception as e:
            logger.error(f"Error merging agency feeds: {e}")
            raise
    
    def process_multiple_agencies(self, agency_urls, city_name):
        """
        Process GTFS feeds from multiple agencies for a city.
        
        Args:
            agency_urls: Dictionary mapping agency names to their GTFS feed URLs
            city_name: Name of the city
            
        Returns:
            Dictionary containing merged stops GeoDataFrame and service frequency DataFrame
        """
        logger.info(f"Processing multiple agencies for {city_name}")
        
        try:
            # Dictionary to store feeds for each agency
            agency_feeds = {}
            
            # Process each agency
            for agency_name, url in agency_urls.items():
                logger.info(f"Processing agency: {agency_name}")
                
                # Create a processor for this agency
                processor = GTFSProcessor(agency_name=agency_name)
                
                # Download the feed
                processor.download_gtfs_feed(agency_url=url)
                
                # Load the feed
                feed = processor.load_gtfs_feed()
                
                # Store in dictionary
                agency_feeds[agency_name] = feed
            
            # Merge feeds
            merged_feed = self.merge_agency_feeds(agency_feeds)
            
            # Extract stops from merged feed
            stops = self.extract_stops(merged_feed)
            
            # Convert to GeoDataFrame
            stops_gdf = self.stops_to_geodataframe(stops)
            
            # Calculate service frequency
            frequency = self.calculate_service_frequency(feed=merged_feed)
            
            # Save results
            output_dir = PROCESSED_DATA_DIR / city_name.lower().replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save stops GeoDataFrame
            stops_file = output_dir / "stops.geojson"
            stops_gdf.to_file(stops_file, driver="GeoJSON")
            
            # Save frequency DataFrame
            frequency_file = output_dir / "frequency.csv"
            frequency.to_csv(frequency_file, index=False)
            
            logger.info(f"Saved processed data for {city_name} to {output_dir}")
            
            return {
                'stops': stops_gdf,
                'frequency': frequency,
                'feed': merged_feed
            }
            
        except Exception as e:
            logger.error(f"Error processing multiple agencies: {e}")
            raise
        
    def save_processed_data(self, data, name, output_dir=None, file_format="geojson"):
        """
        Save processed GTFS data.
        
        Args:
            data: DataFrame or GeoDataFrame to save
            name: Name for the output file (without extension)
            output_dir: Directory to save to (defaults to processed data directory)
            file_format: Format to save as (geojson, csv, parquet)
            
        Returns:
            Path to saved file
        """
        # Use default output directory if none provided
        if output_dir is None:
            if self.agency_name:
                output_dir = PROCESSED_DATA_DIR / self.agency_name
            else:
                output_dir = PROCESSED_DATA_DIR
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file path
        file_path = output_dir / f"{name}.{file_format}"
        
        logger.info(f"Saving processed data to {file_path}")
        
        try:
            # Save based on format
            if file_format == "geojson" and isinstance(data, gpd.GeoDataFrame):
                data.to_file(file_path, driver="GeoJSON")
            elif file_format == "csv":
                data.to_csv(file_path, index=False)
            elif file_format == "parquet":
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def get_agency_info(self, feed=None):
        """
        Get information about the transit agency.
        
        Args:
            feed: Dictionary of GTFS DataFrames (if not already loaded)
            
        Returns:
            DataFrame with agency information
        """
        # Use provided feed or loaded feed
        feed = feed or self.feed
        if not feed:
            raise ValueError("No GTFS feed provided or loaded")
        
        try:
            # Check if agency.txt exists
            if 'agency' in feed:
                return feed['agency']
            else:
                logger.warning("No agency.txt found in GTFS feed")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting agency info: {e}")
            raise