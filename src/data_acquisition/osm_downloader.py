"""
OpenStreetMap data acquisition for urban mobility analysis.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import split, linemerge
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from datetime import datetime
import json
import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up project directory structure
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "osm"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed" / "osm"
CACHE_DIR = PROJECT_DIR / "data" / "interim" / "osm_cache"

# Create data directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx config handled automatically in newer versions

class OSMDownloader:
    """Downloads and processes OpenStreetMap data."""
    
    def __init__(self, place_name=None, boundary=None, cache_folder=None):
        """Initialize with place name or boundary polygon."""
        self.place_name = place_name
        self.boundary = boundary
        self.cache_folder = Path(cache_folder) if cache_folder else CACHE_DIR
        
        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)
        
        if place_name and not boundary:
            try:
                self.boundary = self._get_place_boundary(place_name)
                logger.info(f"Got boundary for {place_name}")
            except Exception as e:
                logger.error(f"Failed to get boundary for {place_name}: {e}")
                raise
    
    def get_street_network(self, boundary=None, network_type='all', simplify=True, cache=True):
        """Download street network within boundary."""
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        place_id = self.place_name or "custom_boundary"
        cache_file = self.cache_folder / f"street_network_{place_id}_{network_type}.graphml"
        
        if cache and cache_file.exists():
            logger.info(f"Loading cached street network from {cache_file}")
            try:
                G = ox.load_graphml(cache_file)
                return G
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Downloading fresh data.")
        
        logger.info(f"Downloading {network_type} street network")
        
        try:
            if isinstance(boundary, gpd.GeoDataFrame):
                boundary_dissolved = boundary.dissolve()
                polygon = boundary_dissolved.iloc[0].geometry
            else:
                polygon = boundary
            
            G = ox.graph_from_polygon(polygon, network_type=network_type, simplify=simplify)
            
            if cache:
                ox.save_graphml(G, cache_file)
                logger.info(f"Cached street network to {cache_file}")
            
            logger.info(f"Downloaded street network: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to download street network: {e}")
            return None
    
    def get_sidewalk_data(self, boundary=None, cache=True):
        """Extract sidewalk and crossing data."""
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        place_id = self.place_name or "custom_boundary"
        cache_file = self.cache_folder / f"sidewalks_{place_id}.geojson"
        
        if cache and cache_file.exists():
            logger.info(f"Loading cached sidewalk data from {cache_file}")
            try:
                gdf = gpd.read_file(cache_file)
                return gdf
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Downloading fresh data.")
        
        logger.info("Downloading sidewalk and crossing data")
        
        try:
            if isinstance(boundary, gpd.GeoDataFrame):
                boundary_dissolved = boundary.dissolve()
                polygon = boundary_dissolved.iloc[0].geometry
            else:
                polygon = boundary
            
            # OSM tags for pedestrian infrastructure
            tags = {
                'highway': ['footway', 'path', 'steps', 'pedestrian', 'crossing'],
                'footway': ['sidewalk', 'crossing'],
                'cycleway': True,
                'sidewalk': True,
                'crossing': True
            }
            
            # Download and clean data
            gdf = ox.features_from_polygon(polygon, tags=tags)
            gdf = gdf[~gdf.geometry.isna()]  # Remove invalid geometries
            gdf['infrastructure_type'] = 'other'
            
            # Categorize sidewalks (pedestrian paths along roads)
            sidewalk_mask = (
                (gdf['highway'] == 'footway') & (gdf['footway'] == 'sidewalk') |
                (gdf['sidewalk'].notna())
            )
            gdf.loc[sidewalk_mask, 'infrastructure_type'] = 'sidewalk'
            
            # Categorize crossings (pedestrian street crossings)
            crossing_mask = (
                (gdf['highway'] == 'footway') & (gdf['footway'] == 'crossing') |
                (gdf['highway'] == 'crossing') |
                (gdf['crossing'].notna())
            )
            gdf.loc[crossing_mask, 'infrastructure_type'] = 'crossing'
            
            # Categorize curb ramps (accessibility features)
            ramp_mask = (
                (gdf['highway'] == 'footway') & (gdf['incline'].notna()) |
                (gdf['kerb'].notna())
            )
            gdf.loc[ramp_mask, 'infrastructure_type'] = 'curb_ramp'
            
            # Save to cache for future use
            if cache:
                gdf.to_file(cache_file, driver="GeoJSON")
                logger.info(f"Cached sidewalk data to {cache_file}")
            
            logger.info(f"Downloaded {len(gdf)} pedestrian infrastructure features")
            return gdf
            
        except Exception as e:
            logger.error(f"Error downloading sidewalk data: {e}")
            # Return empty GeoDataFrame on failure - columns will be added as needed
            return self._create_empty_sidewalk_gdf()
    
    def get_amenities(self, boundary=None, amenity_types=None, cache=True):
        """
        Extract points of interest (amenities) by type within a boundary.
        
        Args:
            boundary: GeoDataFrame or Polygon defining the boundary (overrides self.boundary)
            amenity_types: List of amenity types to extract (e.g., ['school', 'hospital'])
            cache: Whether to use cached data if available
            
        Returns:
            GeoDataFrame of amenities
        """
        # Use provided boundary or instance boundary
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        # Default amenity types if none provided
        if amenity_types is None:
            amenity_types = [
                'school', 'hospital', 'clinic', 'doctors', 'pharmacy',
                'supermarket', 'grocery', 'library', 'community_centre',
                'restaurant', 'cafe', 'bank', 'post_office', 'bus_station'
            ]
        
        # Generate cache file name
        place_id = self.place_name or "custom_boundary"
        amenity_hash = hash(tuple(sorted(amenity_types)))
        cache_file = self.cache_folder / f"amenities_{place_id}_{amenity_hash}.geojson"
        
        # Check if cache exists
        if cache and cache_file.exists():
            logger.info(f"Loading cached amenity data from {cache_file}")
            try:
                gdf = gpd.read_file(cache_file)
                return gdf
            except Exception as e:
                logger.warning(f"Error loading cached amenity data: {e}. Downloading fresh data.")
        
        logger.info(f"Downloading amenity data for types: {amenity_types}")
        
        try:
            # Extract polygon if boundary is a GeoDataFrame
            if isinstance(boundary, gpd.GeoDataFrame):
                # Dissolve to get a single polygon
                boundary_dissolved = boundary.dissolve()
                polygon = boundary_dissolved.iloc[0].geometry
            else:
                polygon = boundary
            
            # Define tags for amenities
            tags = {'amenity': amenity_types}
            
            # Download the data
            gdf = ox.features_from_polygon(polygon, tags=tags)
            
            # Filter for valid geometries
            gdf = gdf[~gdf.geometry.isna()]
            
            # Cache the data
            if cache and not gdf.empty:
                gdf.to_file(cache_file, driver="GeoJSON")
                logger.info(f"Cached amenity data to {cache_file}")
            
            logger.info(f"Downloaded {len(gdf)} amenities")
            return gdf
            
        except Exception as e:
            logger.error(f"Error downloading amenity data: {e}")
            # Return empty GeoDataFrame on failure - columns will be added as needed
            return self._create_empty_amenity_gdf()
    
    def filter_accessible_amenities(self, amenities=None, boundary=None, amenity_types=None):
        """Filter amenities based on accessibility attributes like wheelchair access."""
        # Get amenities if not provided
        if amenities is None:
            amenities = self.get_amenities(boundary, amenity_types)
        
        if amenities is None or amenities.empty:
            logger.warning("No amenities to filter")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326") if amenities is None else amenities
        
        logger.info("Filtering amenities for accessibility")
        
        try:
            accessible = amenities.copy()
            accessible['accessibility_score'] = 0
            
            # Score wheelchair accessibility (0-3 points)
            if 'wheelchair' in accessible.columns:
                wheelchair_scores = {
                    'yes': 3, 'limited': 2, 'designated': 3, 'no': 0, 'unknown': 1
                }
                accessible['wheelchair_score'] = accessible['wheelchair'].map(wheelchair_scores).fillna(1)
                accessible['accessibility_score'] += accessible['wheelchair_score']
            else:
                accessible['wheelchair_score'] = 1  # Unknown
            
            # Add points for other accessibility features
            accessibility_features = [
                'tactile_paving', 'handrail', 'ramp', 'elevator',
                'hearing_impaired', 'visual_impaired'
            ]
            
            for feature in accessibility_features:
                if feature in accessible.columns:
                    # +1 point for each accessibility feature
                    accessible.loc[accessible[feature] == 'yes', 'accessibility_score'] += 1
            
            # Categorize based on total score
            accessible['accessibility_category'] = pd.cut(
                accessible['accessibility_score'],
                bins=[-1, 0, 1, 3, float('inf')],
                labels=['not_accessible', 'unknown', 'partially_accessible', 'fully_accessible']
            )
            
            logger.info(f"Filtered {len(accessible)} amenities for accessibility")
            return accessible
            
        except Exception as e:
            logger.error(f"Failed to filter accessible amenities: {e}")
            raise
        
    def download_by_chunks(self, boundary=None, function_name=None, max_area_km2=50, **kwargs):
        """Download data by splitting large areas into smaller chunks to avoid timeouts."""
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        # Map function names to actual methods
        if function_name == 'get_street_network':
            func = self.get_street_network
        elif function_name == 'get_sidewalk_data':
            func = self.get_sidewalk_data
        elif function_name == 'get_amenities':
            func = self.get_amenities
        else:
            raise ValueError(f"Unknown function name: {function_name}")
        
        # Extract polygon if boundary is a GeoDataFrame
        if isinstance(boundary, gpd.GeoDataFrame):
            # Dissolve to get a single polygon
            boundary_dissolved = boundary.dissolve()
            polygon = boundary_dissolved.iloc[0].geometry
        else:
            polygon = boundary
        
        # Calculate area in square kilometers
        # Convert to projected CRS for accurate area calculation
        boundary_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        boundary_projected = boundary_gdf.to_crs("EPSG:3857")  # Web Mercator
        area_km2 = boundary_projected.area.iloc[0] / 1e6  # Convert from m² to km²
        
        logger.info(f"Boundary area: {area_km2:.2f} km²")
        
        # If area is small enough, download directly
        if area_km2 <= max_area_km2:
            logger.info("Area is small enough, downloading directly")
            return func(boundary=polygon, **kwargs)
        
        logger.info(f"Area exceeds {max_area_km2} km², splitting into chunks")
        
        # Split the polygon into chunks
        chunks = self._split_polygon(polygon, max_area_km2)
        logger.info(f"Split boundary into {len(chunks)} chunks")
        
        # Download data for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                # Create a temporary boundary for this chunk
                chunk_gdf = gpd.GeoDataFrame(geometry=[chunk], crs="EPSG:4326")
                
                # Download data for this chunk
                chunk_result = func(boundary=chunk_gdf, **kwargs)
                
                # Add to results
                if chunk_result is not None:
                    results.append(chunk_result)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with next chunk
        
        # Combine results
        if function_name == 'get_street_network':
            # Combine NetworkX graphs
            if results:
                combined_graph = nx.compose_all(results)
                logger.info(f"Combined graph has {len(combined_graph.nodes)} nodes and {len(combined_graph.edges)} edges")
                return combined_graph
            else:
                logger.warning("No valid results to combine")
                return None
        else:
            # Combine GeoDataFrames
            if results:
                combined_gdf = pd.concat(results, ignore_index=True)
                logger.info(f"Combined GeoDataFrame has {len(combined_gdf)} features")
                return combined_gdf
            else:
                logger.warning("No valid results to combine")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    def _split_polygon(self, polygon, max_area_km2):
        """
        Split a polygon into smaller chunks.
        
        Args:
            polygon: Polygon to split
            max_area_km2: Maximum area of each chunk in square kilometers
            
        Returns:
            List of smaller polygons
        """
        # Convert to projected CRS for accurate area calculation
        boundary_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        boundary_projected = boundary_gdf.to_crs("EPSG:3857")  # Web Mercator
        area_km2 = boundary_projected.area.iloc[0] / 1e6  # Convert from m² to km²
        
        # If area is small enough, return as is
        if area_km2 <= max_area_km2:
            return [polygon]
        
        # Get bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        # Calculate number of divisions needed
        n_divisions = int(np.ceil(np.sqrt(area_km2 / max_area_km2)))
        
        # Create grid
        x_edges = np.linspace(minx, maxx, n_divisions + 1)
        y_edges = np.linspace(miny, maxy, n_divisions + 1)
        
        # Create chunks
        chunks = []
        for i in range(n_divisions):
            for j in range(n_divisions):
                # Create a rectangular chunk
                chunk = Polygon([
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1]),
                    (x_edges[i], y_edges[j])
                ])
                
                # Intersect with original polygon
                intersection = chunk.intersection(polygon)
                
                # Add to chunks if not empty
                if not intersection.is_empty:
                    chunks.append(intersection)
        
        return chunks
    
    def _retry_download(self, func, max_retries=3, initial_delay=5, **kwargs):
        """
        Retry a download function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            **kwargs: Arguments to pass to the function
            
        Returns:
            Result from the function or None if all retries fail
        """
        retries = 0
        delay = initial_delay
        
        while retries < max_retries:
            try:
                logger.info(f"Retry attempt {retries + 1}/{max_retries}")
                return func(**kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Maximum retries reached. Last error: {e}")
                    return None
                
                logger.warning(f"Retry {retries}/{max_retries} failed: {e}")
                logger.info(f"Waiting {delay} seconds before next attempt")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    def _create_empty_sidewalk_gdf(self, crs="EPSG:4326"):
        """
        Create an empty GeoDataFrame for sidewalks with proper structure.
        
        Args:
            crs: Coordinate reference system
            
        Returns:
            Empty GeoDataFrame with sidewalk structure
        """
        # Create empty GeoDataFrame with expected columns
        empty_gdf = gpd.GeoDataFrame({
            'infrastructure_type': [],
            'highway': [],
            'footway': [],
            'sidewalk': [],
            'crossing': [],
            'kerb': [],
            'incline': []
        }, geometry=[], crs=crs)
        return empty_gdf
    
    def _create_empty_amenity_gdf(self, crs="EPSG:4326"):
        """
        Create an empty GeoDataFrame for amenities with proper structure.
        
        Args:
            crs: Coordinate reference system
            
        Returns:
            Empty GeoDataFrame with amenity structure
        """
        # Create empty GeoDataFrame with expected columns
        empty_gdf = gpd.GeoDataFrame({
            'amenity': [],
            'name': [],
            'wheelchair': []
        }, geometry=[], crs=crs)
        return empty_gdf
    
    def _get_place_boundary(self, place_name):
        """
        Get the boundary polygon for a place by name.
        
        Args:
            place_name: Name of the place (e.g., "Seattle, Washington")
            
        Returns:
            GeoDataFrame with the boundary
        """
        try:
            # Try to get the boundary using OSMnx
            gdf = ox.geocode_to_gdf(place_name)
            
            # Check if we got a valid boundary
            if gdf.empty:
                raise ValueError(f"Could not find boundary for {place_name}")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error getting boundary for {place_name}: {e}")
            raise
        
    def calculate_sidewalk_coverage(self, street_network=None, sidewalk_data=None, boundary=None):
        """
        Calculate sidewalk coverage for the street network.
        
        Args:
            street_network: NetworkX graph of the street network (if None, will be downloaded)
            sidewalk_data: GeoDataFrame of sidewalks (if None, will be downloaded)
            boundary: GeoDataFrame or Polygon defining the boundary (overrides self.boundary)
            
        Returns:
            Dictionary with sidewalk coverage statistics
        """
        # Use provided boundary or instance boundary
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        # Download street network if not provided
        if street_network is None:
            street_network = self.get_street_network(boundary, network_type='drive')
        
        # Download sidewalk data if not provided
        if sidewalk_data is None:
            sidewalk_data = self.get_sidewalk_data(boundary)
        
        logger.info("Calculating sidewalk coverage")
        
        try:
            # Convert street network to GeoDataFrame of edges
            streets_gdf = ox.graph_to_gdfs(street_network, nodes=False, edges=True)
            
            # Filter for roads that should have sidewalks (exclude highways, service roads, etc.)
            sidewalk_roads = streets_gdf[
                (streets_gdf['highway'].isin(['residential', 'tertiary', 'secondary', 'primary', 'unclassified'])) &
                (~streets_gdf['highway'].isin(['motorway', 'trunk', 'motorway_link', 'trunk_link', 'service']))
            ]
            
            # Convert to projected CRS for accurate length calculations
            sidewalk_roads_projected = sidewalk_roads.to_crs("EPSG:3857")  # Web Mercator
            
            # Calculate total length of roads that should have sidewalks
            sidewalk_roads_length = sidewalk_roads_projected.length.sum()
            
            # Filter sidewalk data for actual sidewalks (handle missing column gracefully)
            if 'infrastructure_type' in sidewalk_data.columns:
                sidewalks = sidewalk_data[sidewalk_data['infrastructure_type'] == 'sidewalk']
            else:
                # If no infrastructure_type column, assume all are sidewalks
                sidewalks = sidewalk_data
            
            # Convert to projected CRS for accurate length calculations
            if not sidewalks.empty:
                sidewalks_projected = sidewalks.to_crs("EPSG:3857")  # Web Mercator
                sidewalk_length = sidewalks_projected.length.sum()
            else:
                sidewalk_length = 0
            
            # Calculate coverage (considering both sides of the road)
            # A perfect coverage would be 2 * road length (sidewalks on both sides)
            # Cap the coverage at 100% to avoid unrealistic values
            coverage_ratio = min(sidewalk_length / (2 * sidewalk_roads_length), 1.0) if sidewalk_roads_length > 0 else 0
            
            # Count crossings (handle missing column gracefully)
            if 'infrastructure_type' in sidewalk_data.columns:
                crossings = sidewalk_data[sidewalk_data['infrastructure_type'] == 'crossing']
                curb_ramps = sidewalk_data[sidewalk_data['infrastructure_type'] == 'curb_ramp']
            else:
                crossings = gpd.GeoDataFrame(geometry=[], crs=sidewalk_data.crs)
                curb_ramps = gpd.GeoDataFrame(geometry=[], crs=sidewalk_data.crs)
            
            crossing_count = len(crossings) if not crossings.empty else 0
            curb_ramp_count = len(curb_ramps) if not curb_ramps.empty else 0
            
            # Calculate intersection density
            nodes_gdf = ox.graph_to_gdfs(street_network, nodes=True, edges=False)
            intersections = nodes_gdf[nodes_gdf['street_count'] > 1]
            intersection_count = len(intersections)
            
            # Calculate area in square kilometers
            boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326") if not isinstance(boundary, gpd.GeoDataFrame) else boundary
            boundary_projected = boundary_gdf.to_crs("EPSG:3857")  # Web Mercator
            area_km2 = boundary_projected.area.iloc[0] / 1e6  # Convert from m² to km²
            
            # Calculate densities
            intersection_density = intersection_count / area_km2 if area_km2 > 0 else 0
            crossing_density = crossing_count / area_km2 if area_km2 > 0 else 0
            
            # Prepare results
            results = {
                'road_length_km': sidewalk_roads_length / 1000,  # Convert to kilometers
                'sidewalk_length_km': sidewalk_length / 1000,
                'sidewalk_coverage_ratio': coverage_ratio,
                'sidewalk_coverage_percent': coverage_ratio * 100,
                'crossing_count': crossing_count,
                'curb_ramp_count': curb_ramp_count,
                'intersection_count': intersection_count,
                'area_km2': area_km2,
                'intersection_density': intersection_density,
                'crossing_density': crossing_density,
                'crossings_per_intersection': crossing_count / intersection_count if intersection_count > 0 else 0
            }
            
            logger.info(f"Sidewalk coverage: {results['sidewalk_coverage_percent']:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error calculating sidewalk coverage: {e}")
            raise
    
    def analyze_amenity_accessibility(self, amenities=None, sidewalk_data=None, street_network=None, boundary=None):
        """
        Analyze accessibility of amenities based on nearby sidewalks and street network.
        
        Args:
            amenities: GeoDataFrame of amenities (if None, will be downloaded)
            sidewalk_data: GeoDataFrame of sidewalks (if None, will be downloaded)
            street_network: NetworkX graph of the street network (if None, will be downloaded)
            boundary: GeoDataFrame or Polygon defining the boundary (overrides self.boundary)
            
        Returns:
            GeoDataFrame of amenities with accessibility analysis
        """
        # Use provided boundary or instance boundary
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        # Download amenities if not provided
        if amenities is None:
            amenities = self.get_amenities(boundary)
        
        # Download sidewalk data if not provided
        if sidewalk_data is None:
            sidewalk_data = self.get_sidewalk_data(boundary)
        
        # Download street network if not provided
        if street_network is None:
            street_network = self.get_street_network(boundary, network_type='walk')
        
        logger.info("Analyzing amenity accessibility")
        
        try:
            # Filter for actual sidewalks (handle missing column gracefully)
            if 'infrastructure_type' in sidewalk_data.columns:
                sidewalks = sidewalk_data[sidewalk_data['infrastructure_type'] == 'sidewalk']
                crossings = sidewalk_data[sidewalk_data['infrastructure_type'] == 'crossing']
            else:
                # If no infrastructure_type column, assume all are sidewalks
                sidewalks = sidewalk_data
                crossings = gpd.GeoDataFrame(geometry=[], crs=sidewalk_data.crs)
            
            # Create a copy of amenities to avoid modifying the original
            amenities_with_access = amenities.copy()
            
            # Add columns for accessibility analysis
            amenities_with_access['has_nearby_sidewalk'] = False
            amenities_with_access['distance_to_sidewalk'] = float('inf')
            amenities_with_access['has_nearby_crossing'] = False
            amenities_with_access['distance_to_crossing'] = float('inf')
            amenities_with_access['walkability_score'] = 0
            
            # Skip if no amenities
            if amenities_with_access.empty:
                logger.warning("No amenities to analyze")
                return amenities_with_access
            
            # Skip if no sidewalks
            if sidewalks.empty:
                logger.warning("No sidewalks for analysis")
                return amenities_with_access
            
            # Convert to projected CRS for accurate distance calculations
            amenities_projected = amenities_with_access.to_crs("EPSG:3857")  # Web Mercator
            sidewalks_projected = sidewalks.to_crs("EPSG:3857") if not sidewalks.empty else gpd.GeoDataFrame()
            crossings_projected = crossings.to_crs("EPSG:3857") if not crossings.empty else gpd.GeoDataFrame()
            
            # For each amenity, find the nearest sidewalk and crossing
            for idx, amenity in tqdm(amenities_projected.iterrows(), total=len(amenities_projected), desc="Analyzing amenities"):
                # Create a buffer around the amenity for efficient spatial querying
                amenity_buffer = amenity.geometry.buffer(100)  # 100 meter buffer
                
                # Find sidewalks within the buffer
                nearby_sidewalks = sidewalks_projected[sidewalks_projected.intersects(amenity_buffer)]
                if not nearby_sidewalks.empty:
                    # Find nearest sidewalk within buffer
                    nearest_sidewalk_idx = nearby_sidewalks.distance(amenity.geometry).idxmin()
                    nearest_sidewalk_distance = amenity.geometry.distance(nearby_sidewalks.loc[nearest_sidewalk_idx].geometry)
                    
                    # Check if sidewalk is nearby (within 50 meters)
                    if nearest_sidewalk_distance <= 50:
                        amenities_with_access.at[idx, 'has_nearby_sidewalk'] = True
                        amenities_with_access.at[idx, 'distance_to_sidewalk'] = nearest_sidewalk_distance
                
                # Find crossings within the buffer
                if not crossings_projected.empty:
                    nearby_crossings = crossings_projected[crossings_projected.intersects(amenity_buffer)]
                    if not nearby_crossings.empty:
                        # Find nearest crossing within buffer
                        nearest_crossing_idx = nearby_crossings.distance(amenity.geometry).idxmin()
                        nearest_crossing_distance = amenity.geometry.distance(nearby_crossings.loc[nearest_crossing_idx].geometry)
                        
                        # Check if crossing is nearby (within 100 meters)
                        if nearest_crossing_distance <= 100:
                            amenities_with_access.at[idx, 'has_nearby_crossing'] = True
                            amenities_with_access.at[idx, 'distance_to_crossing'] = nearest_crossing_distance
            
            # Calculate walkability score (0-100)
            # 50 points for wheelchair accessibility
            # 30 points for nearby sidewalk
            # 20 points for nearby crossing
            
            # Add points for wheelchair accessibility (handle missing column gracefully)
            if 'wheelchair_score' in amenities_with_access.columns:
                # Scale from 0-3 to 0-50
                amenities_with_access['walkability_score'] += (amenities_with_access['wheelchair_score'] / 3) * 50
            
            # Add points for nearby sidewalk (closer = more points)
            sidewalk_score = 30 * (1 - amenities_with_access['distance_to_sidewalk'] / 50).clip(0, 1)
            # Fix data type warning by converting to float
            amenities_with_access['walkability_score'] = amenities_with_access['walkability_score'].astype(float)
            amenities_with_access.loc[amenities_with_access['has_nearby_sidewalk'], 'walkability_score'] += sidewalk_score[amenities_with_access['has_nearby_sidewalk']]
            
            # Add points for nearby crossing (closer = more points)
            crossing_score = 20 * (1 - amenities_with_access['distance_to_crossing'] / 100).clip(0, 1)
            amenities_with_access.loc[amenities_with_access['has_nearby_crossing'], 'walkability_score'] += crossing_score[amenities_with_access['has_nearby_crossing']]
            
            # Categorize walkability
            amenities_with_access['walkability_category'] = pd.cut(
                amenities_with_access['walkability_score'],
                bins=[0, 25, 50, 75, 100],
                labels=['poor', 'fair', 'good', 'excellent']
            )
            
            logger.info(f"Analyzed accessibility for {len(amenities_with_access)} amenities")
            return amenities_with_access
            
        except Exception as e:
            logger.error(f"Error analyzing amenity accessibility: {e}")
            raise
    
    def save_processed_data(self, data, name, output_dir=None, file_format="geojson"):
        """
        Save processed OSM data.
        
        Args:
            data: DataFrame, GeoDataFrame, or NetworkX graph to save
            name: Name for the output file (without extension)
            output_dir: Directory to save to (defaults to processed data directory)
            file_format: Format to save as (geojson, csv, graphml)
            
        Returns:
            Path to saved file
        """
        # Use default output directory if none provided
        if output_dir is None:
            if self.place_name:
                place_name = self.place_name.lower().replace(", ", "_").replace(" ", "_")
                output_dir = PROCESSED_DATA_DIR / place_name
            else:
                output_dir = PROCESSED_DATA_DIR
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate file format
        valid_formats = ["geojson", "csv", "graphml", "json"]
        if file_format not in valid_formats:
            raise ValueError(f"Unsupported file format: {file_format}. Valid formats: {valid_formats}")
        
        # Generate file path
        file_path = output_dir / f"{name}.{file_format}"
        
        logger.info(f"Saving processed data to {file_path}")
        
        try:
            # Save based on data type and format
            if isinstance(data, nx.Graph) and file_format == "graphml":
                ox.save_graphml(data, file_path)
            elif isinstance(data, gpd.GeoDataFrame) and file_format == "geojson":
                data.to_file(file_path, driver="GeoJSON")
            elif isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)) and file_format == "csv":
                data.to_csv(file_path, index=False)
            elif isinstance(data, dict) and file_format == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported data type or format: {type(data)} / {file_format}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
        
    # Additional methods to add to the OSMDownloader class in osm_downloader.py

    def calculate_isochrones(self, origin_point, travel_times=None, network_type='walk', speed=4.5):
        """
        Calculate isochrones (areas reachable within given times) from a point.
        
        Args:
            origin_point: Point geometry or (lat, lon) tuple
            travel_times: List of travel times in minutes (default: [5, 10, 15])
            network_type: Type of network to use ('walk', 'bike', 'drive', 'all')
            speed: Travel speed in km/h (default: 4.5 km/h for walking)
            
        Returns:
            GeoDataFrame of isochrone polygons
        """
        import pandana
        
        # Default travel times if not provided
        if travel_times is None:
            travel_times = [5, 10, 15]  # 5, 10, and 15 minute walks
        
        logger.info(f"Calculating {travel_times} minute isochrones from origin point")
        
        try:
            # Get the street network
            G = self.get_street_network(network_type=network_type)
            
            # Convert origin point to shapely Point if it's a tuple
            if isinstance(origin_point, tuple):
                origin_point = Point(origin_point[1], origin_point[0])  # (lat, lon) to (x, y)
            
            # Convert NetworkX graph to Pandana network
            nodes, edges = ox.graph_to_gdfs(G)
            
            # Create a Pandana network
            network = pandana.Network(
                nodes['x'],
                nodes['y'],
                edges['u'],
                edges['v'],
                edges[['length']]
            )
            
            # Set the impedance
            network.precompute(travel_times[-1] * 60)  # Convert minutes to seconds
            
            # Find the nearest node to the origin point
            nearest_node = ox.distance.nearest_nodes(G, origin_point.x, origin_point.y)
            
            # Calculate the accessibility
            accessibility = {}
            for time in travel_times:
                # Convert minutes to meters based on walking speed
                distance = (time / 60) * speed * 1000  # km/h to m/s * seconds
                
                # Calculate the accessibility
                accessibility[time] = network.get_node_ids(
                    distance,
                    nearest_node
                )
            
            # Create isochrone polygons
            isochrones = []
            for time in travel_times:
                # Get the nodes within this time
                nodes_in_range = nodes.loc[accessibility[time]]
                
                # Create a convex hull around these nodes
                if len(nodes_in_range) > 2:
                    hull = nodes_in_range.unary_union.convex_hull
                    isochrones.append({
                        'time': time,
                        'geometry': hull
                    })
            
            # Create a GeoDataFrame
            isochrones_gdf = gpd.GeoDataFrame(isochrones, crs="EPSG:4326")
            
            logger.info(f"Created {len(isochrones_gdf)} isochrones")
            return isochrones_gdf
            
        except Exception as e:
            logger.error(f"Error calculating isochrones: {e}")
            raise

    def identify_mobility_barriers(self, street_network=None, sidewalk_data=None, boundary=None):
        """
        Identify potential mobility barriers in the network.
        
        Args:
            street_network: NetworkX graph of the street network (if None, will be downloaded)
            sidewalk_data: GeoDataFrame of sidewalks (if None, will be downloaded)
            boundary: GeoDataFrame or Polygon defining the boundary (overrides self.boundary)
            
        Returns:
            GeoDataFrame of potential mobility barriers
        """
        # Use provided boundary or instance boundary
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        # Download street network if not provided
        if street_network is None:
            street_network = self.get_street_network(boundary, network_type='walk')
        
        # Download sidewalk data if not provided
        if sidewalk_data is None:
            sidewalk_data = self.get_sidewalk_data(boundary)
        
        logger.info("Identifying potential mobility barriers")
        
        try:
            # Convert street network to GeoDataFrame of edges
            streets_gdf = ox.graph_to_gdfs(street_network, nodes=False, edges=True)
            
            # Identify potential barriers
            barriers = []
            
            # 1. Missing sidewalks on major roads
            major_roads = streets_gdf[
                streets_gdf['highway'].isin(['primary', 'secondary', 'tertiary'])
            ]
            
            # Filter for sidewalks (handle missing column gracefully)
            if 'infrastructure_type' in sidewalk_data.columns:
                sidewalks = sidewalk_data[sidewalk_data['infrastructure_type'] == 'sidewalk']
            else:
                # If no infrastructure_type column, assume all are sidewalks
                sidewalks = sidewalk_data
            
            # Convert to projected CRS for accurate distance calculations
            major_roads_projected = major_roads.to_crs("EPSG:3857")  # Web Mercator
            sidewalks_projected = sidewalks.to_crs("EPSG:3857") if not sidewalks.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
            
            # For each major road, check if there's a nearby sidewalk
            for idx, road in major_roads_projected.iterrows():
                # Find nearby sidewalks (within 15 meters)
                nearby_sidewalks = sidewalks_projected[sidewalks_projected.distance(road.geometry) < 15]
                
                # If no nearby sidewalks, this is a potential barrier
                if nearby_sidewalks.empty:
                    # Convert back to geographic coordinates for output
                    road_geo = major_roads.loc[idx]
                    barriers.append({
                        'geometry': road_geo.geometry,
                        'barrier_type': 'missing_sidewalk',
                        'road_type': road_geo['highway'],
                        'name': road_geo.get('name', 'Unnamed road')
                    })
            
            # 2. Crossings without curb ramps (handle missing column gracefully)
            if 'infrastructure_type' in sidewalk_data.columns:
                crossings = sidewalk_data[sidewalk_data['infrastructure_type'] == 'crossing']
                curb_ramps = sidewalk_data[sidewalk_data['infrastructure_type'] == 'curb_ramp']
            else:
                # If no infrastructure_type column, create empty GeoDataFrames with proper structure
                crossings = gpd.GeoDataFrame(geometry=[], crs=sidewalk_data.crs)
                curb_ramps = gpd.GeoDataFrame(geometry=[], crs=sidewalk_data.crs)
            
            # Convert to projected CRS for accurate distance calculations
            crossings_projected = crossings.to_crs("EPSG:3857") if not crossings.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
            curb_ramps_projected = curb_ramps.to_crs("EPSG:3857") if not curb_ramps.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
            
            # For each crossing, check if there's a nearby curb ramp
            for idx, crossing in crossings_projected.iterrows():
                # Find nearby curb ramps (within 10 meters)
                nearby_ramps = curb_ramps_projected[curb_ramps_projected.distance(crossing.geometry) < 10]
                
                # If no nearby curb ramps, this is a potential barrier
                if nearby_ramps.empty:
                    # Convert back to geographic coordinates for output
                    crossing_geo = crossings.loc[idx]
                    barriers.append({
                        'geometry': crossing_geo.geometry,
                        'barrier_type': 'crossing_without_ramp',
                        'crossing_type': crossing_geo.get('crossing', 'unknown'),
                        'name': crossing_geo.get('name', 'Unnamed crossing')
                    })
            
            # 3. Disconnected sidewalk segments (simplified for performance)
            # Skip this analysis for large datasets to avoid O(n²) complexity
            if not sidewalks.empty and len(sidewalks) < 1000:  # Only for smaller areas
                logger.info("Analyzing disconnected sidewalk segments")
                # Create a buffer around each sidewalk end point
                sidewalk_endpoints = []
                for idx, sidewalk in sidewalks.iterrows():
                    if isinstance(sidewalk.geometry, LineString):
                        start_point = Point(sidewalk.geometry.coords[0])
                        end_point = Point(sidewalk.geometry.coords[-1])
                        sidewalk_endpoints.append(start_point.buffer(5))
                        sidewalk_endpoints.append(end_point.buffer(5))
                
                # Find isolated endpoints (not connected to other sidewalks)
                for i, endpoint in enumerate(sidewalk_endpoints):
                    # Skip every other endpoint (to avoid checking both ends of the same sidewalk)
                    if i % 2 == 0:
                        continue
                    
                    # Count nearby endpoints
                    nearby_count = sum(1 for ep in sidewalk_endpoints if endpoint.intersects(ep) and ep != endpoint)
                    
                    # If this endpoint is not connected to any other sidewalk, it's a potential barrier
                    if nearby_count == 0:
                        barriers.append({
                            'geometry': endpoint,
                            'barrier_type': 'disconnected_sidewalk',
                            'name': 'Disconnected sidewalk endpoint'
                        })
            else:
                logger.info(f"Skipping disconnected sidewalk analysis for {len(sidewalks)} sidewalks (too many for efficient processing)")
            
            # Create a GeoDataFrame of barriers
            barriers_gdf = gpd.GeoDataFrame(barriers, crs="EPSG:4326")
            
            logger.info(f"Identified {len(barriers_gdf)} potential mobility barriers")
            return barriers_gdf
            
        except Exception as e:
            logger.error(f"Error identifying mobility barriers: {e}")
            raise

    def assess_osm_data_quality(self, boundary=None):
        """
        Assess the quality and completeness of OSM data in an area.
        
        Args:
            boundary: GeoDataFrame or Polygon defining the boundary (overrides self.boundary)
            
        Returns:
            Dictionary with data quality metrics
        """
        # Use provided boundary or instance boundary
        boundary = boundary if boundary is not None else self.boundary
        if boundary is None:
            raise ValueError("No boundary provided")
        
        logger.info("Assessing OSM data quality")
        
        try:
            # Get street network, sidewalks, and amenities
            street_network = self.get_street_network(boundary=boundary)
            sidewalk_data = self.get_sidewalk_data(boundary=boundary)
            amenities = self.get_amenities(boundary=boundary)
            
            # Calculate metrics
            metrics = {}
            
            # 1. Street network density
            nodes, edges = ox.graph_to_gdfs(street_network)
            
            # Calculate area in square kilometers
            boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326") if not isinstance(boundary, gpd.GeoDataFrame) else boundary
            boundary_projected = boundary_gdf.to_crs("EPSG:3857")  # Web Mercator
            area_km2 = boundary_projected.area.iloc[0] / 1e6  # Convert from m² to km²
            
            metrics['area_km2'] = area_km2
            metrics['node_count'] = len(nodes)
            metrics['edge_count'] = len(edges)
            metrics['node_density'] = len(nodes) / area_km2
            metrics['edge_density'] = len(edges) / area_km2
            
            # 2. Sidewalk completeness
            sidewalks = sidewalk_data[sidewalk_data['infrastructure_type'] == 'sidewalk'] if not sidewalk_data.empty else gpd.GeoDataFrame()
            crossings = sidewalk_data[sidewalk_data['infrastructure_type'] == 'crossing'] if not sidewalk_data.empty else gpd.GeoDataFrame()
            
            metrics['sidewalk_count'] = len(sidewalks)
            metrics['crossing_count'] = len(crossings)
            metrics['sidewalk_density'] = len(sidewalks) / area_km2 if area_km2 > 0 else 0
            metrics['crossing_density'] = len(crossings) / area_km2 if area_km2 > 0 else 0
            
            # 3. Amenity completeness
            metrics['amenity_count'] = len(amenities)
            metrics['amenity_density'] = len(amenities) / area_km2 if area_km2 > 0 else 0
            
            # 4. Accessibility tagging completeness
            if not amenities.empty and 'wheelchair' in amenities.columns:
                wheelchair_tagged = amenities['wheelchair'].notna().sum()
                metrics['wheelchair_tagging_rate'] = (wheelchair_tagged / len(amenities)) * 100
            else:
                metrics['wheelchair_tagging_rate'] = 0
            
            # 5. Compare to expected values for urban areas
            # These are rough benchmarks based on typical urban areas
            expected_node_density = 200  # nodes per km²
            expected_sidewalk_density = 15  # sidewalks per km²
            expected_crossing_density = 10  # crossings per km²
            expected_amenity_density = 20  # amenities per km²
            
            metrics['node_density_score'] = min(100, (metrics['node_density'] / expected_node_density) * 100)
            metrics['sidewalk_density_score'] = min(100, (metrics['sidewalk_density'] / expected_sidewalk_density) * 100)
            metrics['crossing_density_score'] = min(100, (metrics['crossing_density'] / expected_crossing_density) * 100)
            metrics['amenity_density_score'] = min(100, (metrics['amenity_density'] / expected_amenity_density) * 100)
            
            # Overall quality score (0-100)
            metrics['overall_quality_score'] = (
                metrics['node_density_score'] * 0.2 +
                metrics['sidewalk_density_score'] * 0.3 +
                metrics['crossing_density_score'] * 0.3 +
                metrics['amenity_density_score'] * 0.1 +
                metrics['wheelchair_tagging_rate'] * 0.1
            )
            
            logger.info(f"OSM data quality assessment complete. Overall score: {metrics['overall_quality_score']:.1f}/100")
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing OSM data quality: {e}")
            raise