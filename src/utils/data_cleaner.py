# src/utils/data_cleaner.py
"""
Data cleaning and integration for the urban mobility project.

This module handles fetching, cleaning, and combining data from multiple sources:
- Census demographic data
- Transit stop data (GTFS)
- OpenStreetMap infrastructure data

It standardizes coordinate systems, validates data, and creates integrated datasets
for mobility analysis.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import warnings
import time
import requests
from functools import wraps
from shapely.geometry import Point
from typing import Dict, Any, Optional, Callable

# Import project utilities
from .spatial_utils import (
    ensure_crs, safe_spatial_join, validate_and_repair_geometries,
    clip_to_boundary, calculate_area, calculate_length
)
from .data_validator import DataValidator
from .config_manager import ConfigManager
from .data_persistence import DataPersistence
from .data_quality import DataQualityAssessor

# Import data acquisition modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_acquisition.fetch_census_data import CensusFetcher
from src.data_acquisition.gtfs_processor import GTFSProcessor
from src.data_acquisition.osm_downloader import OSMDownloader

# Import analysis modules
from src.analysis.transit_score import TransitScoreCalculator, calculate_comprehensive_transit_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry functions on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def handle_api_failure(func: Callable) -> Callable:
    """
    Decorator to handle API failures gracefully.
    
    Args:
        func: Function to wrap with API error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed in {func.__name__}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limit exceeded in {func.__name__}. Consider implementing rate limiting.")
            elif e.response.status_code >= 500:  # Server error
                logger.error(f"Server error in {func.__name__}: {e.response.status_code}")
            else:
                logger.error(f"HTTP error in {func.__name__}: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper

class DataCleaner:
    """Handles data fetching, cleaning, and integration for mobility analysis."""
    
    def __init__(self, config_path=None):
        """
        Initialize the data cleaner.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize components
        self.config = ConfigManager(config_path)
        self.validator = DataValidator()
        self.persistence = DataPersistence(base_dir=self.config.processed_dir)
        self.quality_assessor = DataQualityAssessor()
        
        # Set coordinate systems
        self.default_crs = self.config.get("crs.default", "EPSG:4326")
        self.analysis_crs = self.config.get("crs.analysis", "EPSG:3857")
        
        # Data containers
        self.census_data = None
        self.transit_data = None
        self.osm_data = {
            'street_network': None,
            'sidewalks': None,
            'amenities': None
        }
        self.integrated_data = {}
        
        # Track data lineage
        self.data_lineage = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'transformations': [],
            'metadata': {
                'pipeline_version': '1.0.0',
                'run_timestamp': datetime.now().isoformat(),
                'city': None,
                'total_sources': 0,
                'total_transformations': 0
            }
        }
        
        # Error tracking
        self.errors = []
        self.warnings = []
    
    def add_error(self, source: str, error: str, details: Optional[Dict] = None):
        """Add an error to the error tracking list."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'error': error,
            'details': details or {}
        }
        self.errors.append(error_entry)
        logger.error(f"[{source}] {error}")
    
    def add_warning(self, source: str, warning: str, details: Optional[Dict] = None):
        """Add a warning to the warning tracking list."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'warning': warning,
            'details': details or {}
        }
        self.warnings.append(warning_entry)
        logger.warning(f"[{source}] {warning}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors and warnings."""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def handle_missing_data(self, data_type: str, fallback_strategy: str = 'skip') -> bool:
        """
        Handle missing data gracefully.
        
        Args:
            data_type: Type of data that is missing
            fallback_strategy: Strategy to use ('skip', 'placeholder', 'error')
            
        Returns:
            True if handled successfully, False otherwise
        """
        if fallback_strategy == 'skip':
            self.add_warning('data_cleaner', f"Missing {data_type} data - skipping")
            return True
        elif fallback_strategy == 'placeholder':
            self.add_warning('data_cleaner', f"Missing {data_type} data - using placeholder")
            # Create placeholder data based on type
            if data_type == 'transit':
                self.transit_data = self._create_placeholder_transit_data()
            elif data_type == 'census':
                self.census_data = self._create_placeholder_census_data()
            return True
        elif fallback_strategy == 'error':
            self.add_error('data_cleaner', f"Missing {data_type} data - cannot proceed")
            return False
        return False
    
    def _create_placeholder_transit_data(self) -> gpd.GeoDataFrame:
        """Create placeholder transit data when real data is unavailable."""
        from shapely.geometry import Point
        
        placeholder_data = gpd.GeoDataFrame(
            {
                'stop_id': ['placeholder_1', 'placeholder_2'],
                'stop_name': ['Placeholder Stop 1', 'Placeholder Stop 2'],
                'stop_lat': [47.6062, 47.6097],
                'stop_lon': [-122.3321, -122.3331],
                'wheelchair_accessible': ['yes', 'yes'],
                'trips_per_day': [10, 10],
                'agency_name': ['placeholder_agency', 'placeholder_agency']
            },
            geometry=[Point(-122.3321, 47.6062), Point(-122.3331, 47.6097)],
            crs="EPSG:4326"
        )
        return placeholder_data
    
    def _create_placeholder_census_data(self) -> gpd.GeoDataFrame:
        """Create placeholder census data when real data is unavailable."""
        from shapely.geometry import Polygon
        
        # Create a simple polygon for Seattle area
        polygon = Polygon([
            (-122.5, 47.5), (-122.3, 47.5), (-122.3, 47.7), (-122.5, 47.7), (-122.5, 47.5)
        ])
        
        placeholder_data = gpd.GeoDataFrame(
            {
                'geoid': ['53033000000'],
                'total_population': [1000],
                'median_household_income': [50000],
                'area_sqkm': [100.0]
            },
            geometry=[polygon],
            crs="EPSG:4326"
        )
        return placeholder_data
    
    def fetch_all_data(self, city_name, force_refresh=False):
        """
        Fetch data from all sources for a city.
        
        Args:
            city_name: Name of the city
            force_refresh: Whether to force refresh all data
            
        Returns:
            True if successful, False otherwise
        """
        city_config = self.config.get_city_config(city_name)
        if not city_config:
            logger.error(f"City configuration not found for {city_name}")
            return False
        
        try:
            # Fetch each data source
            sources = [
                ('Census', self.fetch_census_data, {'state': city_config['state'], 'county': city_config['county']}),
                ('Transit', self.fetch_transit_data, {'city_name': city_name}),
                ('OSM', self.fetch_osm_data, {'place_name': f"{city_name}, {city_config['state']}"})
            ]
            
            for source_name, fetch_func, args in sources:
                logger.info(f"Fetching {source_name} data for {city_name}")
                success = fetch_func(force_refresh=force_refresh, **args)
                if not success:
                    logger.warning(f"{source_name} data fetch failed")
            
            logger.info(f"Data fetching complete for {city_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return False
    
    def fetch_census_data(self, state, county, force_refresh=False):
        """
        Fetch Census data for a county.
        
        Args:
            state: State FIPS code or name
            county: County FIPS code or name
            force_refresh: Whether to force refresh data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for existing data
            if not force_refresh:
                existing = self.persistence.get_latest_version("census_tracts", category="census")
                if existing:
                    logger.info(f"Loading existing Census data from {existing['file_path']}")
                    self.census_data = self.persistence.load_dataframe(file_path=existing['file_path'])
                    self._add_lineage('census', 'cache', existing['file_path'], existing['timestamp'])
                    return True
            
            # Fetch new data
            api_key = self.config.get("api_keys.census")
            fetcher = CensusFetcher(api_key=api_key)
            
            boundaries = fetcher.get_census_boundaries(state, county)
            demographics = fetcher.get_demographic_data(state=state, county=county)
            self.census_data = fetcher.merge_census_data(boundaries, demographics)
            
            # Validate and save
            is_valid, issues = fetcher.validate_census_data(self.census_data)
            if not is_valid:
                logger.warning(f"Census data validation issues: {issues}")
            
            file_path = self.persistence.save_dataframe(
                self.census_data,
                name="census_tracts",
                category="census",
                format="geoparquet",
                crs=self.default_crs,
                metadata={'state': state, 'county': county, 'validation': issues if not is_valid else 'passed'}
            )
            
            self._add_lineage('census', 'api', str(file_path), datetime.now().isoformat(), {'state': state, 'county': county})
            
            logger.info(f"Census data processed: {len(self.census_data)} tracts")
            
            # Assess data quality
            self.assess_data_quality('census', self.census_data)
            
            return True
            
        except Exception as e:
            self.add_error('census_fetch', f"Failed to fetch Census data: {e}", {
                'state': state,
                'county': county,
                'error_type': type(e).__name__
            })
            return False
    
    def fetch_transit_data(self, city_name, force_refresh=False):
        """
        Fetch transit data for a city.
        
        Args:
            city_name: Name of the city
            force_refresh: Whether to force refresh data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for existing data
            if not force_refresh:
                existing = self.persistence.get_latest_version("transit_stops", category="transit")
                if existing:
                    logger.info(f"Loading existing transit data from {existing['file_path']}")
                    self.transit_data = self.persistence.load_dataframe(file_path=existing['file_path'])
                    self._add_lineage('transit', 'cache', existing['file_path'], existing['timestamp'])
                    return True
            
            # Get city config
            city_config = self.config.get_city_config(city_name)
            if not city_config:
                self.add_error('transit_fetch', f"City configuration not found for {city_name}")
                return False
            
            # Check for GTFS data - look for agency directories
            gtfs_base_dir = self.config.raw_dir / "gtfs"
            if not gtfs_base_dir.exists():
                self.add_error('transit_fetch', f"GTFS base directory not found: {gtfs_base_dir}")
                return False
            
            # Look for agency directories (king_county_metro, sound_transit, etc.)
            agency_dirs = [d for d in gtfs_base_dir.iterdir() if d.is_dir() and any(d.glob("*.txt"))]
            
            if not agency_dirs:
                self.add_error('transit_fetch', f"No GTFS agency directories found in {gtfs_base_dir}")
                return False
            
            logger.info(f"Found GTFS data for {len(agency_dirs)} agencies: {[d.name for d in agency_dirs]}")
            
            # Process all agencies and combine their data
            all_transit_data = []
            for agency_dir in agency_dirs:
                try:
                    logger.info(f"Processing GTFS data from {agency_dir}")
                    agency_data = self._process_gtfs_data(agency_dir)
                    if agency_data is not None and not agency_data.empty:
                        # Add agency identifier
                        agency_data['agency_name'] = agency_dir.name
                        all_transit_data.append(agency_data)
                        logger.info(f"Processed {len(agency_data)} stops from {agency_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to process {agency_dir.name}: {e}")
            
            if not all_transit_data:
                logger.error("No transit data could be processed from any agency")
                return False
            
            # Combine all agency data
            self.transit_data = pd.concat(all_transit_data, ignore_index=True)
            self._add_lineage('transit', 'gtfs', str(gtfs_base_dir), datetime.now().isoformat(), 
                            {'city': city_name, 'agencies': [d.name for d in agency_dirs]})
            
            # Save processed data
            file_path = self.persistence.save_dataframe(
                self.transit_data,
                name="transit_stops",
                category="transit",
                format="geoparquet",
                crs=self.default_crs,
                metadata={'city': city_name, 'stop_count': len(self.transit_data)}
            )
            
            logger.info(f"Transit data processed: {len(self.transit_data)} stops")
            
            # Assess data quality
            self.assess_data_quality('transit', self.transit_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch transit data: {e}")
            return False
    
    def fetch_osm_data(self, place_name, force_refresh=False):
        """
        Fetch OSM data for a place.
        
        Args:
            place_name: Name of the place
            force_refresh: Whether to force refresh data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for existing data
            if not force_refresh:
                existing_network = self.persistence.get_latest_version("street_network", category="osm")
                existing_sidewalks = self.persistence.get_latest_version("sidewalks", category="osm")
                existing_amenities = self.persistence.get_latest_version("amenities", category="osm")
                
                if existing_network and existing_sidewalks and existing_amenities:
                    logger.info("Loading existing OSM data")
                    self._load_existing_osm_data(existing_network, existing_sidewalks, existing_amenities)
                    return True
            
            # Download new data
            downloader = OSMDownloader(place_name=place_name)
            
            logger.info("Downloading OSM data")
            self.osm_data['street_network'] = downloader.get_street_network(network_type='all')
            self.osm_data['sidewalks'] = downloader.get_sidewalk_data()
            amenities = downloader.get_amenities()
            self.osm_data['amenities'] = downloader.filter_accessible_amenities(amenities)
            
            # Save data
            place_slug = place_name.lower().replace(", ", "_").replace(" ", "_")
            self._save_osm_data(place_slug, place_name)
            
            self._add_lineage('osm', 'api', None, datetime.now().isoformat(), {'place': place_name})
            
            logger.info("OSM data fetched and processed")
            
            # Assess data quality for OSM datasets
            for osm_type, osm_data in self.osm_data.items():
                if osm_data is not None:
                    self.assess_data_quality(f'osm_{osm_type}', osm_data)
            
            return True
            
        except Exception as e:
            self.add_error('osm_fetch', f"Failed to fetch OSM data: {e}", {
                'place_name': place_name,
                'error_type': type(e).__name__
            })
            return False
    
    def integrate_data(self, city_name):
        """
        Integrate data from all sources into a mobility index.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Dictionary of integrated datasets
        """
        logger.info(f"Integrating data for {city_name}")
        
        try:
            # Check data availability
            if not self._check_data_availability():
                return None
            
            # Standardize and clean data
            census_data, transit_data, sidewalks, amenities = self._prepare_data_for_integration()
            self._add_transformation('data_preparation', 'prepared_datasets', 'Prepared and standardized all datasets')
            
            # Calculate mobility metrics
            census_data = self._calculate_transit_metrics(census_data, transit_data)
            self._add_transformation('transit_metrics', 'transit_scores', 'Calculated transit access metrics')
            
            census_data = self._calculate_sidewalk_metrics(census_data, sidewalks)
            
            # Calculate comprehensive sidewalk scores if crossings data is available
            crossings = None
            curb_ramps = None
            
            # Try to extract crossings and curb ramps from sidewalks data
            if 'infrastructure_type' in sidewalks.columns:
                crossings = sidewalks[sidewalks['infrastructure_type'] == 'crossing'].copy()
                curb_ramps = sidewalks[sidewalks['infrastructure_type'] == 'curb_ramp'].copy()
            
            census_data = self._calculate_comprehensive_sidewalk_scores(
                census_data, sidewalks, crossings, curb_ramps
            )
            self._add_transformation('sidewalk_metrics', 'sidewalk_scores', 'Calculated sidewalk quality metrics')
            
            # Calculate comprehensive amenity scores if amenities data is available
            census_data = self._calculate_comprehensive_amenity_scores(census_data, amenities)
            self._add_transformation('amenity_metrics', 'amenity_scores', 'Calculated comprehensive amenity proximity metrics')
            
            # Calculate mobility index
            census_data = self._calculate_mobility_index(census_data)
            self._add_transformation('mobility_index_calculation', 'mobility_scores', 'Calculated composite Mobility Accessibility Index')
            
            # Store and save results
            self._store_integrated_data(census_data, transit_data, sidewalks, amenities)
            
            integrated_path = self.persistence.save_dataframe(
                census_data,
                name="mobility_index",
                category="integrated",
                format="geoparquet",
                crs=self.default_crs,
                metadata={
                    'city': city_name,
                    'integration_date': datetime.now().isoformat(),
                    'tract_count': len(census_data)
                }
            )
            
            self._add_transformation('data_integration', str(integrated_path))
            
            logger.info(f"Data integration complete for {city_name}")
            
            # Generate comprehensive quality report
            quality_report = self.generate_quality_report()
            self._add_transformation('quality_assessment', 'quality_report', 'Generated comprehensive quality assessment report')
            logger.info(f"Overall data quality score: {quality_report.get('overall_quality_score', 0.0):.2f}")
            
            # Save error report if there are any errors or warnings
            if self.errors or self.warnings:
                self.save_error_report()
                logger.info(f"Error report saved: {len(self.errors)} errors, {len(self.warnings)} warnings")
            
            return self.integrated_data
            
        except Exception as e:
            self.add_error('data_integration', f"Failed to integrate data: {e}", {
                'city_name': city_name,
                'error_type': type(e).__name__
            })
            return None
    
    def save_data_lineage(self, output_file=None):
        """
        Save data lineage information.
        
        Args:
            output_file: Path to save lineage information
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = self.config.processed_dir / "data_lineage.json"
        
        # Update metadata with final counts
        self.data_lineage['metadata']['total_sources'] = len(self.data_lineage['sources'])
        self.data_lineage['metadata']['total_transformations'] = len(self.data_lineage['transformations'])
        self.data_lineage['metadata']['run_timestamp'] = datetime.now().isoformat()
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.data_lineage, f, indent=2)
            
            logger.info(f"Saved data lineage to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to save data lineage: {e}")
            return self.data_lineage
    
    def save_error_report(self, output_file=None):
        """
        Save error and warning report.
        
        Args:
            output_file: Path to save error report
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = self.config.processed_dir / "error_report.json"
        
        try:
            error_summary = self.get_error_summary()
            with open(output_file, 'w') as f:
                json.dump(error_summary, f, indent=2)
            
            logger.info(f"Saved error report to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
            return None
    
    def create_basic_visualization(self, output_dir=None):
        """
        Create basic visualizations of the integrated data.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualizations
        """
        if output_dir is None:
            output_dir = self.config.processed_dir / "visualizations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.integrated_data or 'census_with_mobility' not in self.integrated_data:
            logger.warning("No integrated data available. Call integrate_data first.")
            return []
        
        try:
            import matplotlib.pyplot as plt
            data = self.integrated_data['census_with_mobility']
            
            # Create mobility scores visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot mobility scores
            scores = [
                ('mobility_access_index', 'Mobility Access Index', 'viridis'),
                ('transit_access_score', 'Transit Access Score', 'Blues'),
                ('sidewalk_quality_score', 'Sidewalk Quality Score', 'Greens'),
                ('amenity_proximity_score', 'Amenity Proximity Score', 'Reds')
            ]
            
            for i, (col, title, cmap) in enumerate(scores):
                row, col_idx = i // 2, i % 2
                data.plot(column=col, ax=axes[row, col_idx], legend=True, 
                         cmap=cmap, legend_kwds={'label': title})
                axes[row, col_idx].set_title(f"{title} by Census Tract")
            
            plt.tight_layout()
            scores_path = output_dir / "mobility_scores.png"
            plt.savefig(scores_path, dpi=300)
            plt.close()
            
            # Create relationship plots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Income vs transit access
            if 'median_household_income' in data.columns:
                axes[0].scatter(data['median_household_income'], data['transit_access_score'], alpha=0.5)
                axes[0].set_xlabel('Median Household Income ($)')
                axes[0].set_ylabel('Transit Access Score')
                axes[0].set_title('Transit Access vs. Median Income')
                
                # Add trend line
                if len(data) > 1:
                    try:
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            data['median_household_income'].fillna(0), 
                            data['transit_access_score'].fillna(0)
                        )
                        x = np.array([data['median_household_income'].min(), data['median_household_income'].max()])
                        axes[0].plot(x, intercept + slope * x, 'r', label=f'R² = {r_value**2:.2f}')
                        axes[0].legend()
                    except Exception as e:
                        logger.warning(f"Error adding trend line: {e}")
            
            # Mobility index distribution
            data['mobility_access_index'].plot.hist(ax=axes[1], bins=20)
            axes[1].set_xlabel('Mobility Access Index')
            axes[1].set_ylabel('Number of Census Tracts')
            axes[1].set_title('Distribution of Mobility Access Index')
            
            plt.tight_layout()
            relationships_path = output_dir / "mobility_relationships.png"
            plt.savefig(relationships_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved visualizations to {output_dir}")
            return [scores_path, relationships_path]
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return []
    
    # Helper methods
    
    def _process_gtfs_data(self, gtfs_dir):
        """Process GTFS data and extract stops with frequency information."""
        processor = GTFSProcessor()
        feed = processor.load_gtfs_feed(gtfs_dir)
        stops = processor.extract_stops(feed)
        transit_data = processor.stops_to_geodataframe(stops)
        
        # Add frequency information
        frequency = processor.calculate_service_frequency(feed=feed)
        if not frequency.empty:
            transit_data = transit_data.merge(
                frequency[['stop_id', 'trips_per_day', 'trips_per_hour', 'avg_headway_minutes']],
                on='stop_id',
                how='left'
            )
        
        return transit_data
    
    def _load_existing_osm_data(self, existing_network, existing_sidewalks, existing_amenities):
        """Load existing OSM data from files."""
        # Load street network
        try:
            import osmnx as ox
            self.osm_data['street_network'] = ox.load_graphml(existing_network['file_path'])
        except Exception as e:
            logger.warning(f"Error loading street network: {e}")
        
        # Load sidewalks and amenities
        self.osm_data['sidewalks'] = self.persistence.load_dataframe(file_path=existing_sidewalks['file_path'])
        self.osm_data['amenities'] = self.persistence.load_dataframe(file_path=existing_amenities['file_path'])
        
        self._add_lineage('osm', 'cache', existing_network['file_path'], existing_network['timestamp'])
    
    def _save_osm_data(self, place_slug, place_name):
        """Save OSM data to files."""
        # Save street network
        try:
            import osmnx as ox
            network_path = self.config.processed_dir / "osm" / f"{place_slug}_street_network.graphml"
            network_path.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(self.osm_data['street_network'], network_path)
            logger.info(f"Saved street network to {network_path}")
        except Exception as e:
            logger.warning(f"Error saving street network: {e}")
        
        # Save sidewalks and amenities
        self.persistence.save_dataframe(
            self.osm_data['sidewalks'],
            name="sidewalks",
            category="osm",
            format="geoparquet",
            crs=self.default_crs,
            metadata={'place': place_name, 'feature_count': len(self.osm_data['sidewalks'])}
        )
        
        self.persistence.save_dataframe(
            self.osm_data['amenities'],
            name="amenities",
            category="osm",
            format="geoparquet",
            crs=self.default_crs,
            metadata={'place': place_name, 'feature_count': len(self.osm_data['amenities'])}
        )
    
    def _check_data_availability(self):
        """Check if all required datasets are loaded."""
        if self.census_data is None:
            logger.warning("Census data not loaded. Call fetch_census_data first.")
            return False
        
        if self.transit_data is None:
            logger.warning("Transit data not loaded. Call fetch_transit_data first.")
            return False
        
        if self.osm_data['street_network'] is None or self.osm_data['sidewalks'] is None or self.osm_data['amenities'] is None:
            logger.warning("OSM data not loaded. Call fetch_osm_data first.")
            return False
        
        return True
    
    def _prepare_data_for_integration(self):
        """Standardize and clean data for integration."""
        logger.info("Preparing data for integration")
        
        # Standardize CRS
        census_data = ensure_crs(self.census_data, self.default_crs)
        transit_data = ensure_crs(self.transit_data, self.default_crs)
        sidewalks = ensure_crs(self.osm_data['sidewalks'], self.default_crs)
        amenities = ensure_crs(self.osm_data['amenities'], self.default_crs)
        
        # Validate and repair geometries
        census_data = validate_and_repair_geometries(census_data)
        transit_data = validate_and_repair_geometries(transit_data)
        sidewalks = validate_and_repair_geometries(sidewalks)
        amenities = validate_and_repair_geometries(amenities)
        
        # Calculate areas and lengths
        census_data = calculate_area(census_data, 'area_sqkm')
        sidewalks = calculate_length(sidewalks, 'length_km')
        
        return census_data, transit_data, sidewalks, amenities
    
    def _calculate_transit_metrics(self, census_data, transit_data):
        """Calculate transit access metrics per census tract using new scoring system."""
        logger.info("Calculating comprehensive transit metrics using new scoring system")
        
        try:
            # Initialize transit score calculator
            transit_calculator = TransitScoreCalculator(
                max_walking_distance=self.config.get("processing.max_distance_m", 1000),
                weights=self.config.get("processing.transit_weights", {
                    'distance': 0.4,
                    'frequency': 0.3,
                    'accessibility': 0.2,
                    'coverage': 0.1
                })
            )
            
            # Prepare service data from transit stops
            service_data = None
            if 'trips_per_day' in transit_data.columns:
                service_cols = ['stop_id', 'trips_per_day', 'trips_per_hour', 'avg_headway_minutes']
                available_cols = ['stop_id'] + [col for col in service_cols[1:] if col in transit_data.columns]
                service_data = transit_data[available_cols].copy()
            
            # Calculate comprehensive transit scores
            census_with_scores = transit_calculator.calculate_comprehensive_transit_score(
                neighborhoods=census_data,
                transit_stops=transit_data,
                service_data=service_data,
                distance_method='euclidean'
            )
            
            # Also calculate the legacy metrics for backward compatibility
            census_with_scores = self._calculate_legacy_transit_metrics(census_with_scores, transit_data)
            
            logger.info("Transit metrics calculation complete using new scoring system")
            return census_with_scores
            
        except Exception as e:
            logger.error(f"Error calculating transit metrics with new system: {e}")
            logger.info("Falling back to legacy transit metrics calculation")
            return self._calculate_legacy_transit_metrics(census_data, transit_data)
    
    def _calculate_legacy_transit_metrics(self, census_data, transit_data):
        """Calculate legacy transit metrics for backward compatibility."""
        logger.info("Calculating legacy transit metrics")
        
        # Join transit stops to census tracts
        transit_per_tract = safe_spatial_join(
            transit_data,
            census_data[['geoid', 'geometry']],
            how="inner",
            predicate="within"
        )
        
        if not transit_per_tract.empty:
            # Count stops per tract and calculate service metrics
            transit_metrics = transit_per_tract.groupby('geoid').agg({
                'stop_id': 'count',  # Count of stops
                'trips_per_day': 'sum',  # Total trips per day
                'trips_per_hour': 'sum',  # Total trips per hour
                'avg_headway_minutes': 'mean',  # Average headway
                'wheelchair_accessible': lambda x: (x == 'yes').sum()  # Count accessible stops
            }).reset_index()
            
            # Rename columns
            transit_metrics.columns = ['geoid', 'stop_count', 'total_trips_per_day', 'total_trips_per_hour', 'avg_headway_minutes', 'accessible_stop_count']
            
            # Calculate density and accessibility percentage
            transit_metrics = transit_metrics.merge(
                census_data[['geoid', 'area_sqkm']],
                on='geoid',
                how='left'
            )
            transit_metrics['stops_per_sqkm'] = transit_metrics['stop_count'] / transit_metrics['area_sqkm']
            transit_metrics['pct_accessible_stops'] = (transit_metrics['accessible_stop_count'] / transit_metrics['stop_count']) * 100
            
            # Merge back to census data
            census_data = census_data.merge(
                transit_metrics[['geoid', 'stop_count', 'stops_per_sqkm', 'total_trips_per_day', 'total_trips_per_hour', 'avg_headway_minutes', 'accessible_stop_count', 'pct_accessible_stops']],
                on='geoid',
                how='left'
            )
            
            # Fill NaN values
            transit_cols = ['stop_count', 'stops_per_sqkm', 'total_trips_per_day', 'total_trips_per_hour', 'avg_headway_minutes', 'accessible_stop_count', 'pct_accessible_stops']
            for col in transit_cols:
                census_data[col] = census_data[col].fillna(0)
        else:
            # No transit data - set all transit metrics to 0
            transit_cols = ['stop_count', 'stops_per_sqkm', 'total_trips_per_day', 'total_trips_per_hour', 'avg_headway_minutes', 'accessible_stop_count', 'pct_accessible_stops']
            for col in transit_cols:
                census_data[col] = 0
        
        return census_data
    
    def _calculate_sidewalk_metrics(self, census_data, sidewalks):
        """Calculate sidewalk coverage metrics per census tract."""
        logger.info("Calculating sidewalk metrics")
        
        # Join sidewalks to census tracts
        sidewalks_per_tract = safe_spatial_join(
            sidewalks,
            census_data[['geoid', 'geometry']],
            how="inner",
            predicate="intersects"
        )
        
        if not sidewalks_per_tract.empty:
            # Sum sidewalk lengths per tract
            sidewalk_length = sidewalks_per_tract.groupby('geoid')['length_km'].sum().reset_index()
            
            # Merge back to census data
            census_data = census_data.merge(
                sidewalk_length,
                on='geoid',
                how='left'
            )
            
            # Calculate density
            census_data['length_km'] = census_data['length_km'].fillna(0)
            census_data['sidewalk_km_per_sqkm'] = census_data['length_km'] / census_data['area_sqkm']
        else:
            census_data['length_km'] = 0
            census_data['sidewalk_km_per_sqkm'] = 0
        
        return census_data
    
    def _calculate_comprehensive_sidewalk_scores(self, census_data, sidewalks, crossings=None, curb_ramps=None):
        """Calculate comprehensive sidewalk scores using the new scoring system."""
        try:
            if sidewalks is None or sidewalks.empty:
                logger.warning("No sidewalk data available for comprehensive scoring")
                return census_data
            
            # Import the sidewalk scoring module
            from src.analysis.sidewalk_score import SidewalkScoreCalculator
            
            # Initialize calculator
            calculator = SidewalkScoreCalculator()
            
            # Calculate comprehensive scores
            logger.info(f"Calculating comprehensive sidewalk scores for {len(census_data)} neighborhoods")
            result = calculator.calculate_comprehensive_sidewalk_score(
                neighborhoods=census_data,
                sidewalks=sidewalks,
                crossings=crossings,
                curb_ramps=curb_ramps
            )
            logger.info(f"Comprehensive scoring complete. Result columns: {list(result.columns)}")
            
            # Update census_data with new scores
            score_columns = [
                'sidewalk_quality_score', 'coverage_score', 'ramp_score', 
                'island_score', 'accessibility_score'
            ]
            
            for col in score_columns:
                if col in result.columns:
                    census_data[col] = result[col]
            
            logger.info("Comprehensive sidewalk scores calculation complete")
            return census_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive sidewalk scores: {e}")
            return census_data
    
    def _calculate_amenity_metrics(self, census_data, amenities):
        """Calculate amenity access metrics per census tract."""
        logger.info("Calculating amenity metrics")
        
        # Join amenities to census tracts
        amenities_per_tract = safe_spatial_join(
            amenities,
            census_data[['geoid', 'geometry']],
            how="inner",
            predicate="within"
        )
        
        if not amenities_per_tract.empty:
            # Count amenities per tract
            amenity_counts = amenities_per_tract.groupby('geoid').size().reset_index(name='amenity_count')
            
            # Count accessible amenities if available
            if 'accessibility_category' in amenities_per_tract.columns:
                accessible = amenities_per_tract[amenities_per_tract['accessibility_category'] == 'fully_accessible']
                if not accessible.empty:
                    accessible_counts = accessible.groupby('geoid').size().reset_index(name='accessible_amenity_count')
                    amenity_counts = amenity_counts.merge(accessible_counts, on='geoid', how='left')
                    amenity_counts['accessible_amenity_count'] = amenity_counts['accessible_amenity_count'].fillna(0)
                    amenity_counts['pct_accessible_amenities'] = (
                        amenity_counts['accessible_amenity_count'] / amenity_counts['amenity_count']
                    ) * 100
                else:
                    amenity_counts['accessible_amenity_count'] = 0
                    amenity_counts['pct_accessible_amenities'] = 0
            
            # Merge back to census data
            census_data = census_data.merge(amenity_counts, on='geoid', how='left')
            
            # Fill NaN values and calculate density
            census_data['amenity_count'] = census_data['amenity_count'].fillna(0)
            if 'accessible_amenity_count' in census_data.columns:
                census_data['accessible_amenity_count'] = census_data['accessible_amenity_count'].fillna(0)
                census_data['pct_accessible_amenities'] = census_data['pct_accessible_amenities'].fillna(0)
            census_data['amenities_per_sqkm'] = census_data['amenity_count'] / census_data['area_sqkm']
        else:
            census_data['amenity_count'] = 0
            census_data['accessible_amenity_count'] = 0
            census_data['pct_accessible_amenities'] = 0
            census_data['amenities_per_sqkm'] = 0
        
        return census_data
    
    def _calculate_comprehensive_amenity_scores(self, census_data, amenities):
        """Calculate comprehensive amenity proximity scores using the new scoring system."""
        try:
            if amenities is None or amenities.empty:
                logger.warning("No amenity data available for comprehensive scoring")
                return census_data
            
            # Import the amenity scoring module
            from src.analysis.amenity_score import AmenityScoreCalculator
            
            # Initialize calculator
            amenity_calculator = AmenityScoreCalculator(
                max_distance=2000.0,  # 2km max distance
                distance_method='euclidean',
                accessibility_penalty=0.5
            )
            
            logger.info("Calculating comprehensive amenity scores for all neighborhoods")
            
            # Calculate comprehensive amenity scores
            census_data = amenity_calculator.calculate_comprehensive_amenity_score(
                census_data, amenities
            )
            
            logger.info("Comprehensive amenity scoring complete. Result columns: " + 
                       str([col for col in census_data.columns if 'amenity' in col.lower()]))
            
            return census_data
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive amenity scores: {e}")
            # Fall back to basic amenity metrics
            return self._calculate_amenity_metrics(census_data, amenities)
    
    def assess_data_quality(self, dataset_name: str, data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Assess quality of a dataset and add to lineage.
        
        Args:
            dataset_name: Name of the dataset
            data: GeoDataFrame to assess
            
        Returns:
            Quality metrics dictionary
        """
        logger.info(f"Assessing quality for {dataset_name}")
        
        # Assess quality based on dataset type
        if dataset_name == 'census':
            metrics = self.quality_assessor.assess_census_data_quality(data)
        elif dataset_name == 'transit':
            metrics = self.quality_assessor.assess_transit_data_quality(data)
        elif dataset_name == 'mobility_index':
            metrics = self.quality_assessor.assess_mobility_index_quality(data)
        elif dataset_name.startswith('osm_'):
            osm_type = dataset_name.replace('osm_', '')
            metrics = self.quality_assessor.assess_osm_data_quality(data, osm_type)
        else:
            # Generic assessment
            metrics = {
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'overall_score': 0.0,
                'completeness': {'total_rows': len(data) if data is not None else 0}
            }
        
        # Add quality metrics to lineage
        self._add_lineage(dataset_name, 'quality_assessment', None, datetime.now().isoformat(), 
                         {'quality_score': metrics.get('overall_score', 0.0)})
        
        return metrics
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report for all datasets.
        
        Returns:
            Quality report dictionary
        """
        logger.info("Generating comprehensive quality report")
        
        all_metrics = {}
        
        # Assess quality for each dataset
        if self.census_data is not None:
            all_metrics['census'] = self.assess_data_quality('census', self.census_data)
        
        if self.transit_data is not None:
            all_metrics['transit'] = self.assess_data_quality('transit', self.transit_data)
        
        if 'census_with_mobility' in self.integrated_data:
            all_metrics['mobility_index'] = self.assess_data_quality('mobility_index', self.integrated_data['census_with_mobility'])
        
        for osm_type, osm_data in self.osm_data.items():
            if osm_data is not None:
                all_metrics[f'osm_{osm_type}'] = self.assess_data_quality(f'osm_{osm_type}', osm_data)
        
        # Generate comprehensive report
        report = self.quality_assessor.generate_quality_report(all_metrics)
        
        # Save quality report
        report_file = self.config.processed_dir / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {report_file}")
        
        return report
    
    def _calculate_mobility_index(self, census_data):
        """Calculate the Mobility Accessibility Index."""
        logger.info("Calculating Mobility Accessibility Index")
        
        # Get weights from config
        weights = self.config.get("processing.mai_weights", {
            "transit_access": 0.25,
            "sidewalk_quality": 0.25,
            "amenity_proximity": 0.25,
            "street_connectivity": 0.25
        })
        
        # Calculate component scores (0-100 scale)
        
        # Use new comprehensive transit access score if available, otherwise fall back to legacy calculation
        if 'transit_access_score' in census_data.columns:
            logger.info("Using comprehensive transit access scores from new scoring system")
            # The new scoring system already provides a 0-100 scale score
            transit_score = census_data['transit_access_score']
        else:
            logger.info("Using legacy transit score calculation")
            # Enhanced transit access score - consider multiple transit metrics
            transit_score = 0
            if 'stops_per_sqkm' in census_data.columns:
                # Stop density component (40% weight)
                max_stops = census_data['stops_per_sqkm'].max()
                if max_stops > 0:
                    stop_density_score = (census_data['stops_per_sqkm'] / max_stops) * 40
                else:
                    stop_density_score = 0
                
                # Service frequency component (30% weight)
                if 'total_trips_per_day' in census_data.columns:
                    max_trips = census_data['total_trips_per_day'].max()
                    if max_trips > 0:
                        frequency_score = (census_data['total_trips_per_day'] / max_trips) * 30
                    else:
                        frequency_score = 0
                else:
                    frequency_score = 0
                
                # Accessibility component (20% weight)
                if 'accessible_stops_per_sqkm' in census_data.columns:
                    max_accessible = census_data['accessible_stops_per_sqkm'].max()
                    if max_accessible > 0:
                        accessibility_score = (census_data['accessible_stops_per_sqkm'] / max_accessible) * 20
                    else:
                        accessibility_score = 0
                else:
                    accessibility_score = 0
                
                # Service quality component (10% weight) - inverse of headway (shorter headway = better)
                if 'avg_headway_minutes' in census_data.columns:
                    # Normalize headway (lower is better, so invert)
                    min_headway = census_data['avg_headway_minutes'].min()
                    max_headway = census_data['avg_headway_minutes'].max()
                    if max_headway > min_headway:
                        headway_score = ((max_headway - census_data['avg_headway_minutes']) / (max_headway - min_headway)) * 10
                    else:
                        headway_score = 0
                else:
                    headway_score = 0
                
                # Combine all transit components
                transit_score = stop_density_score + frequency_score + accessibility_score + headway_score
                
                # Store individual components for analysis
                census_data['transit_density_score'] = stop_density_score
                census_data['transit_frequency_score'] = frequency_score
                census_data['transit_accessibility_score'] = accessibility_score
                census_data['transit_quality_score'] = headway_score
            
            census_data['transit_access_score'] = transit_score
        
        # Sidewalk quality score - use new comprehensive scoring if available
        if 'sidewalk_quality_score' not in census_data.columns:
            logger.info("Comprehensive sidewalk scoring not found, using fallback method")
            if 'sidewalk_km_per_sqkm' in census_data.columns:
                max_sidewalk = census_data['sidewalk_km_per_sqkm'].max()
                if max_sidewalk > 0:
                    census_data['sidewalk_quality_score'] = (census_data['sidewalk_km_per_sqkm'] / max_sidewalk) * 100
                else:
                    census_data['sidewalk_quality_score'] = 0
            else:
                census_data['sidewalk_quality_score'] = 0
        
        # Amenity proximity score
        if 'amenities_per_sqkm' in census_data.columns:
            max_amenities = census_data['amenities_per_sqkm'].max()
            if max_amenities > 0:
                census_data['amenity_proximity_score'] = (census_data['amenities_per_sqkm'] / max_amenities) * 100
            else:
                census_data['amenity_proximity_score'] = 0
        else:
            census_data['amenity_proximity_score'] = 0
        
        # Street connectivity score - use intersection density or network connectivity
        # For now, use a combination of sidewalk density and amenity density as proxy
        if 'sidewalk_km_per_sqkm' in census_data.columns and 'amenities_per_sqkm' in census_data.columns:
            # Normalize both metrics and take average
            sidewalk_norm = census_data['sidewalk_km_per_sqkm'] / census_data['sidewalk_km_per_sqkm'].max() if census_data['sidewalk_km_per_sqkm'].max() > 0 else 0
            amenity_norm = census_data['amenities_per_sqkm'] / census_data['amenities_per_sqkm'].max() if census_data['amenities_per_sqkm'].max() > 0 else 0
            census_data['street_connectivity_score'] = ((sidewalk_norm + amenity_norm) / 2) * 100
        else:
            census_data['street_connectivity_score'] = 0
        
        # Calculate weighted MAI
        census_data['mobility_access_index'] = (
            (weights['transit_access'] * census_data['transit_access_score']) +
            (weights['sidewalk_quality'] * census_data['sidewalk_quality_score']) +
            (weights['amenity_proximity'] * census_data['amenity_proximity_score']) +
            (weights['street_connectivity'] * census_data['street_connectivity_score'])
        )
        
        # Normalize to 0-1 range
        max_index = census_data['mobility_access_index'].max()
        if max_index > 0:
            census_data['mobility_access_index'] = census_data['mobility_access_index'] / max_index
        
        return census_data
    
    def _store_integrated_data(self, census_data, transit_data, sidewalks, amenities):
        """Store integrated datasets."""
        logger.info("Storing integrated datasets")
        
        self.integrated_data['census_with_mobility'] = census_data
        
        # Store additional datasets if they exist
        if not transit_data.empty:
            self.integrated_data['transit_data'] = transit_data
        if not sidewalks.empty:
            self.integrated_data['sidewalks'] = sidewalks
        if not amenities.empty:
            self.integrated_data['amenities'] = amenities
    
    def _add_lineage(self, source, type, file_path, timestamp, metadata=None):
        """Add source to data lineage."""
        self.data_lineage['sources'][source] = {
            'source': type,
            'timestamp': timestamp
        }
        if file_path:
            self.data_lineage['sources'][source]['file_path'] = file_path
        if metadata:
            self.data_lineage['sources'][source].update(metadata)
    
    def _add_transformation(self, name, output, description=None):
        """Add transformation to data lineage."""
        self.data_lineage['transformations'].append({
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'output': output,
            'description': description or f'Transformation: {name}'
        })