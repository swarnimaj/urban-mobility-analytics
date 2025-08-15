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
from shapely.geometry import Point

# Import project utilities
from .spatial_utils import (
    ensure_crs, safe_spatial_join, validate_and_repair_geometries,
    clip_to_boundary, calculate_area, calculate_length
)
from .data_validator import DataValidator
from .config_manager import ConfigManager
from .data_persistence import DataPersistence

# Import data acquisition modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_acquisition.fetch_census_data import CensusFetcher
from src.data_acquisition.gtfs_processor import GTFSProcessor
from src.data_acquisition.osm_downloader import OSMDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            'transformations': []
        }
    
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
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch Census data: {e}")
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
                logger.error(f"City configuration not found for {city_name}")
                return False
            
            # Check for GTFS data
            gtfs_dir = self.config.raw_dir / "gtfs" / city_name.lower().replace(" ", "_")
            if not gtfs_dir.exists() or not any(gtfs_dir.iterdir()):
                logger.warning(f"No GTFS data found for {city_name}, creating placeholder data")
                self.transit_data = self._create_placeholder_transit_data(city_config)
                self._add_lineage('transit', 'placeholder', None, datetime.now().isoformat(), {'city': city_name})
            else:
                logger.info(f"Processing GTFS data from {gtfs_dir}")
                self.transit_data = self._process_gtfs_data(gtfs_dir)
                self._add_lineage('transit', 'gtfs', str(gtfs_dir), datetime.now().isoformat(), {'city': city_name})
            
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
            return True
            
        except Exception as e:
            logger.error(f"Failed to fetch OSM data: {e}")
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
            
            # Calculate mobility metrics
            census_data = self._calculate_transit_metrics(census_data, transit_data)
            census_data = self._calculate_sidewalk_metrics(census_data, sidewalks)
            census_data = self._calculate_amenity_metrics(census_data, amenities)
            
            # Calculate mobility index
            census_data = self._calculate_mobility_index(census_data)
            
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
            return self.integrated_data
            
        except Exception as e:
            logger.error(f"Failed to integrate data: {e}")
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
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.data_lineage, f, indent=2)
            
            logger.info(f"Saved data lineage to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to save data lineage: {e}")
            return self.data_lineage
    
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
    def _create_placeholder_transit_data(self, city_config):
        """Create placeholder transit data when GTFS is not available."""
        if 'bbox' in city_config:
            center_lon = (city_config['bbox'][0] + city_config['bbox'][2]) / 2
            center_lat = (city_config['bbox'][1] + city_config['bbox'][3]) / 2
        else:
            center_lon, center_lat = -122.3321, 47.6062  # Seattle center
        
        return gpd.GeoDataFrame(
            {
                'stop_id': ['sample1', 'sample2', 'sample3'],
                'stop_name': ['Sample Stop 1', 'Sample Stop 2', 'Sample Stop 3'],
                'wheelchair_accessible': ['yes', 'no', 'unknown'],
                'trips_per_day': [100, 50, 25],
                'trips_per_hour': [6, 3, 1.5],
                'avg_headway_minutes': [10, 20, 40]
            },
            geometry=[
                Point(center_lon - 0.01, center_lat - 0.01),
                Point(center_lon + 0.01, center_lat - 0.01),
                Point(center_lon, center_lat + 0.01)
            ],
            crs=self.default_crs
        )
    
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
        """Calculate transit access metrics per census tract."""
        logger.info("Calculating transit metrics")
        
        # Join transit stops to census tracts
        transit_per_tract = safe_spatial_join(
            transit_data,
            census_data[['geoid', 'geometry']],
            how="inner",
            predicate="within"
        )
        
        if not transit_per_tract.empty:
            # Count stops per tract
            stops_per_tract = transit_per_tract.groupby('geoid').size().reset_index(name='stop_count')
            
            # Calculate density
            stops_per_tract = stops_per_tract.merge(
                census_data[['geoid', 'area_sqkm']],
                on='geoid',
                how='left'
            )
            stops_per_tract['stops_per_sqkm'] = stops_per_tract['stop_count'] / stops_per_tract['area_sqkm']
            
            # Merge back to census data
            census_data = census_data.merge(
                stops_per_tract[['geoid', 'stop_count', 'stops_per_sqkm']],
                on='geoid',
                how='left'
            )
            
            # Fill NaN values
            census_data['stop_count'] = census_data['stop_count'].fillna(0)
            census_data['stops_per_sqkm'] = census_data['stops_per_sqkm'].fillna(0)
        else:
            census_data['stop_count'] = 0
            census_data['stops_per_sqkm'] = 0
        
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
        score_columns = [
            ('stops_per_sqkm', 'transit_access_score'),
            ('sidewalk_km_per_sqkm', 'sidewalk_quality_score'),
            ('amenities_per_sqkm', 'amenity_proximity_score')
        ]
        
        for input_col, score_col in score_columns:
            if input_col in census_data.columns:
                max_val = census_data[input_col].max()
                if max_val > 0:
                    census_data[score_col] = (census_data[input_col] / max_val) * 100
                else:
                    census_data[score_col] = 0
            else:
                census_data[score_col] = 0
        
        # Street connectivity score (placeholder - uses amenity score for now)
        census_data['street_connectivity_score'] = census_data['amenity_proximity_score']
        
        # Calculate weighted MAI
        census_data['mobility_access_index'] = (
            (weights['transit_access'] * census_data['transit_access_score']) +
            (weights['sidewalk_quality'] * census_data['sidewalk_quality_score']) +
            (weights['amenity_proximity'] * census_data['amenity_proximity_score']) +
            (weights['street_connectivity'] * census_data['street_connectivity_score'])
        )
        
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