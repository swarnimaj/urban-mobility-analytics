# src/analysis/transit_score.py
"""
Transit Access Score Module for Urban Mobility Analytics.

This module calculates comprehensive transit accessibility scores for neighborhoods
based on distance to transit stops, service frequency, and accessibility features.

The scoring system considers:
1. Distance to nearest transit stops (multiple calculation methods)
2. Service frequency and quality (trips per day, headway)
3. Wheelchair accessibility of transit stops
4. Transfer opportunities and network connectivity
5. Service coverage across different times of day

Functions:
- calculate_distance_to_transit(): Calculate distances using multiple methods
- weight_by_frequency(): Weight scores by service frequency
- adjust_for_accessibility(): Adjust scores for wheelchair accessibility
- normalize_transit_score(): Scale scores to 0-100 range
- calculate_comprehensive_transit_score(): Main scoring function
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union
import warnings
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
# Optional import for enhanced normalization
try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = None

# Import project utilities
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.spatial_utils import (
    ensure_crs, calculate_distances, create_buffers, 
    validate_and_repair_geometries, DEFAULT_ANALYSIS_CRS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_DISTANCE = 1000  # meters
DEFAULT_WALK_SPEED = 4.5  # km/h
DEFAULT_WEIGHTS = {
    'distance': 0.4,
    'frequency': 0.3,
    'accessibility': 0.2,
    'coverage': 0.1
}

class TransitScoreCalculator:
    """
    Main class for calculating transit accessibility scores.
    """
    
    def __init__(self, 
                 max_walking_distance: float = DEFAULT_MAX_DISTANCE,
                 walking_speed: float = DEFAULT_WALK_SPEED,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the transit score calculator.
        
        Args:
            max_walking_distance: Maximum walking distance to transit (meters)
            walking_speed: Average walking speed (km/h)
            weights: Dictionary of scoring component weights
        """
        self.max_walking_distance = max_walking_distance
        self.walking_speed = walking_speed
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        
        # Validate weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {sum(self.weights.values())}")
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Initialized TransitScoreCalculator with max distance: {max_walking_distance}m")
    
    def calculate_distance_to_transit(self, 
                                    neighborhoods: gpd.GeoDataFrame,
                                    transit_stops: gpd.GeoDataFrame,
                                    method: str = 'euclidean',
                                    **kwargs) -> gpd.GeoDataFrame:
        """
        Calculate distances from neighborhoods to transit stops using various methods.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons/centroids
            transit_stops: GeoDataFrame of transit stop points
            method: Distance calculation method ('euclidean', 'walking_network', 'multiple_stops')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            GeoDataFrame with distance metrics added
        """
        if neighborhoods is None or neighborhoods.empty:
            logger.error("No neighborhoods provided")
            return neighborhoods
        
        if transit_stops is None or transit_stops.empty:
            logger.warning("No transit stops provided")
            result = neighborhoods.copy()
            result['nearest_stop_distance'] = float('inf')
            result['stop_count_within_buffer'] = 0
            return result
        
        logger.info(f"Calculating transit distances using {method} method")
        
        # Ensure both datasets have consistent CRS
        neighborhoods = ensure_crs(neighborhoods, DEFAULT_ANALYSIS_CRS)
        transit_stops = ensure_crs(transit_stops, DEFAULT_ANALYSIS_CRS)
        
        # Get neighborhood centroids if they're polygons
        if neighborhoods.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            neighborhood_centroids = neighborhoods.copy()
            neighborhood_centroids.geometry = neighborhoods.geometry.centroid
        else:
            neighborhood_centroids = neighborhoods.copy()
        
        result = neighborhoods.copy()
        
        if method == 'euclidean':
            result = self._calculate_euclidean_distance(neighborhood_centroids, transit_stops, result)
        elif method == 'walking_network':
            result = self._calculate_network_distance(neighborhood_centroids, transit_stops, result, **kwargs)
        elif method == 'multiple_stops':
            result = self._calculate_multiple_stops_access(neighborhood_centroids, transit_stops, result, **kwargs)
        else:
            logger.error(f"Unknown distance method: {method}")
            result['nearest_stop_distance'] = float('inf')
        
        return result
    
    def _calculate_euclidean_distance(self, 
                                    neighborhoods: gpd.GeoDataFrame,
                                    transit_stops: gpd.GeoDataFrame,
                                    result: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate straight-line distance to nearest transit stop."""
        try:
            # Calculate distances to all stops for each neighborhood
            distances_df = []
            
            for idx, neighborhood in neighborhoods.iterrows():
                # Calculate distance to all stops
                distances = transit_stops.geometry.distance(neighborhood.geometry)
                
                if not distances.empty:
                    min_distance = distances.min()
                    nearest_stop_idx = distances.idxmin()
                    
                    # Count stops within walking distance
                    stops_within_range = (distances <= self.max_walking_distance).sum()
                    
                    distances_df.append({
                        'neighborhood_idx': idx,
                        'nearest_stop_distance': min_distance,
                        'nearest_stop_id': transit_stops.loc[nearest_stop_idx, 'stop_id'] if 'stop_id' in transit_stops.columns else nearest_stop_idx,
                        'stop_count_within_buffer': stops_within_range
                    })
                else:
                    distances_df.append({
                        'neighborhood_idx': idx,
                        'nearest_stop_distance': float('inf'),
                        'nearest_stop_id': None,
                        'stop_count_within_buffer': 0
                    })
            
            # Merge results back
            distances_df = pd.DataFrame(distances_df).set_index('neighborhood_idx')
            for col in distances_df.columns:
                result[col] = distances_df[col]
            
            logger.info("Euclidean distance calculation complete")
            return result
            
        except Exception as e:
            logger.error(f"Euclidean distance calculation failed: {e}")
            result['nearest_stop_distance'] = float('inf')
            result['stop_count_within_buffer'] = 0
            return result
    
    def _calculate_network_distance(self, 
                                  neighborhoods: gpd.GeoDataFrame,
                                  transit_stops: gpd.GeoDataFrame,
                                  result: gpd.GeoDataFrame,
                                  street_network: Optional[object] = None) -> gpd.GeoDataFrame:
        """
        Calculate network-based walking distance to transit stops.
        
        Note: This is a simplified implementation. For full network routing,
        consider using libraries like pandana, osmnx, or networkx.
        """
        try:
            if street_network is None:
                logger.warning("No street network provided, falling back to euclidean distance with penalty")
                # Apply a penalty factor to euclidean distance to approximate network distance
                result = self._calculate_euclidean_distance(neighborhoods, transit_stops, result)
                # Network distance is typically 1.3-1.5x euclidean distance in urban areas
                result['nearest_stop_distance'] *= 1.35
                return result
            
            # If we have a street network, implement proper network routing
            # This would require additional dependencies like networkx or pandana
            logger.info("Network distance calculation would require street network routing")
            logger.info("Using euclidean distance with network penalty factor for now")
            
            result = self._calculate_euclidean_distance(neighborhoods, transit_stops, result)
            result['nearest_stop_distance'] *= 1.35  # Network penalty factor
            
            return result
            
        except Exception as e:
            logger.error(f"Network distance calculation failed: {e}")
            return self._calculate_euclidean_distance(neighborhoods, transit_stops, result)
    
    def _calculate_multiple_stops_access(self, 
                                       neighborhoods: gpd.GeoDataFrame,
                                       transit_stops: gpd.GeoDataFrame,
                                       result: gpd.GeoDataFrame,
                                       buffer_sizes: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """Calculate access to multiple stops within different distance buffers."""
        try:
            if buffer_sizes is None:
                buffer_sizes = [200, 400, 800, 1200]  # meters
            
            for buffer_size in buffer_sizes:
                col_name = f'stops_within_{buffer_size}m'
                result[col_name] = 0
                
                # Create buffers around neighborhoods
                neighborhood_buffers = create_buffers(neighborhoods, buffer_size)
                
                # Count stops within each buffer
                for idx, neighborhood_buffer in neighborhood_buffers.iterrows():
                    stops_in_buffer = transit_stops[
                        transit_stops.geometry.within(neighborhood_buffer.geometry)
                    ]
                    result.loc[idx, col_name] = len(stops_in_buffer)
            
            # Calculate primary distance metric
            result = self._calculate_euclidean_distance(neighborhoods, transit_stops, result)
            
            logger.info("Multiple stops access calculation complete")
            return result
            
        except Exception as e:
            logger.error(f"Multiple stops access calculation failed: {e}")
            return self._calculate_euclidean_distance(neighborhoods, transit_stops, result)
    
    def weight_by_frequency(self, 
                          neighborhoods: gpd.GeoDataFrame,
                          transit_stops: gpd.GeoDataFrame,
                          service_data: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """
        Weight transit access scores by service frequency.
        
        Args:
            neighborhoods: GeoDataFrame with distance calculations
            transit_stops: GeoDataFrame of transit stops
            service_data: DataFrame with service frequency data
            
        Returns:
            GeoDataFrame with frequency-weighted scores
        """
        try:
            logger.info("Calculating frequency-weighted transit scores")
            
            result = neighborhoods.copy()
            
            # Prepare service frequency data
            if service_data is None:
                # Use service data from transit_stops if available
                frequency_cols = ['trips_per_day', 'trips_per_hour', 'avg_headway_minutes']
                available_cols = [col for col in frequency_cols if col in transit_stops.columns]
                
                if not available_cols:
                    logger.warning("No service frequency data available, using uniform frequency")
                    result['frequency_score'] = 50.0  # Neutral score
                    return result
                
                service_data = transit_stops[['stop_id'] + available_cols].copy()
            
            # Calculate frequency scores for each neighborhood
            frequency_scores = []
            
            for idx, neighborhood in result.iterrows():
                if 'stop_count_within_buffer' in neighborhood and neighborhood['stop_count_within_buffer'] > 0:
                    # Get nearby stops within walking distance
                    nearby_stops = self._get_nearby_stops(
                        neighborhood, transit_stops, self.max_walking_distance
                    )
                    
                    if not nearby_stops.empty:
                        # Calculate weighted frequency score
                        freq_score = self._calculate_frequency_score(nearby_stops, service_data)
                        frequency_scores.append(freq_score)
                    else:
                        frequency_scores.append(0.0)
                else:
                    frequency_scores.append(0.0)
            
            result['frequency_score'] = frequency_scores
            
            # Normalize frequency scores to 0-100 range
            if len(frequency_scores) > 0 and max(frequency_scores) > 0:
                if SKLEARN_AVAILABLE and MinMaxScaler is not None:
                    scaler = MinMaxScaler(feature_range=(0, 100))
                    result['frequency_score'] = scaler.fit_transform(
                        np.array(frequency_scores).reshape(-1, 1)
                    ).flatten()
                else:
                    # Manual min-max scaling
                    min_score = min(frequency_scores)
                    max_score = max(frequency_scores)
                    if max_score > min_score:
                        result['frequency_score'] = [
                            100 * (score - min_score) / (max_score - min_score) 
                            for score in frequency_scores
                        ]
                    else:
                        result['frequency_score'] = [50.0] * len(frequency_scores)
            
            logger.info("Frequency weighting complete")
            return result
            
        except Exception as e:
            logger.error(f"Frequency weighting failed: {e}")
            result['frequency_score'] = 0.0
            return result
    
    def _get_nearby_stops(self, 
                         neighborhood: pd.Series,
                         transit_stops: gpd.GeoDataFrame,
                         max_distance: float) -> gpd.GeoDataFrame:
        """Get transit stops within walking distance of a neighborhood."""
        try:
            # Create buffer around neighborhood
            if hasattr(neighborhood, 'geometry'):
                neighborhood_buffer = neighborhood.geometry.buffer(max_distance)
                nearby_stops = transit_stops[
                    transit_stops.geometry.within(neighborhood_buffer)
                ]
                return nearby_stops
            else:
                return gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Error getting nearby stops: {e}")
            return gpd.GeoDataFrame()
    
    def _calculate_frequency_score(self, 
                                 nearby_stops: gpd.GeoDataFrame,
                                 service_data: pd.DataFrame) -> float:
        """Calculate frequency score for nearby transit stops."""
        try:
            # Merge stop data with service frequency data
            if 'stop_id' in nearby_stops.columns and 'stop_id' in service_data.columns:
                stops_with_service = nearby_stops.merge(
                    service_data, on='stop_id', how='left'
                )
            else:
                stops_with_service = nearby_stops.copy()
            
            # Calculate frequency metrics
            total_trips_per_day = 0
            avg_headway = []
            
            if 'trips_per_day' in stops_with_service.columns:
                total_trips_per_day = stops_with_service['trips_per_day'].fillna(0).sum()
            
            if 'avg_headway_minutes' in stops_with_service.columns:
                headway_values = stops_with_service['avg_headway_minutes'].dropna()
                if not headway_values.empty:
                    avg_headway = headway_values.mean()
                else:
                    avg_headway = 60  # Default 60-minute headway if no data
            else:
                avg_headway = 60
            
            # Convert to frequency score (0-100)
            # High trips per day and low headway = high score
            trips_score = min(100, (total_trips_per_day / 100) * 10)  # Scale factor
            headway_score = max(0, 100 - avg_headway)  # Lower headway = higher score
            
            # Combine scores (weight trips more heavily)
            frequency_score = (trips_score * 0.7) + (headway_score * 0.3)
            
            return frequency_score
            
        except Exception as e:
            logger.error(f"Error calculating frequency score: {e}")
            return 0.0
    
    def adjust_for_accessibility(self, 
                               neighborhoods: gpd.GeoDataFrame,
                               transit_stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Adjust transit scores based on wheelchair accessibility.
        
        Args:
            neighborhoods: GeoDataFrame with transit scores
            transit_stops: GeoDataFrame with accessibility information
            
        Returns:
            GeoDataFrame with accessibility-adjusted scores
        """
        try:
            logger.info("Adjusting scores for wheelchair accessibility")
            
            result = neighborhoods.copy()
            accessibility_scores = []
            
            for idx, neighborhood in result.iterrows():
                # Get nearby stops
                nearby_stops = self._get_nearby_stops(
                    neighborhood, transit_stops, self.max_walking_distance
                )
                
                if not nearby_stops.empty and 'wheelchair_accessible' in nearby_stops.columns:
                    # Calculate accessibility metrics
                    total_stops = len(nearby_stops)
                    accessible_stops = (nearby_stops['wheelchair_accessible'] == 'yes').sum()
                    
                    if total_stops > 0:
                        accessibility_ratio = accessible_stops / total_stops
                        # Convert to 0-100 score
                        accessibility_score = accessibility_ratio * 100
                    else:
                        accessibility_score = 0.0
                else:
                    # No accessibility data available
                    accessibility_score = 50.0  # Neutral score
                
                accessibility_scores.append(accessibility_score)
            
            result['accessibility_score'] = accessibility_scores
            
            logger.info("Accessibility adjustment complete")
            return result
            
        except Exception as e:
            logger.error(f"Accessibility adjustment failed: {e}")
            result['accessibility_score'] = 50.0
            return result
    
    def normalize_transit_score(self, 
                              raw_scores: Union[pd.Series, np.ndarray],
                              method: str = 'minmax',
                              target_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
        """
        Normalize raw transit scores to a specified range.
        
        Args:
            raw_scores: Raw scores to normalize
            method: Normalization method ('minmax', 'zscore', 'robust')
            target_range: Target range for normalized scores
            
        Returns:
            Normalized scores array
        """
        try:
            scores_array = np.array(raw_scores)
            
            # Handle edge cases
            if len(scores_array) == 0:
                return scores_array
            
            if np.all(np.isnan(scores_array)) or np.all(np.isinf(scores_array)):
                logger.warning("All scores are NaN or infinite")
                return np.full_like(scores_array, target_range[0])
            
            # Remove infinite values for normalization
            finite_mask = np.isfinite(scores_array)
            if not np.any(finite_mask):
                return np.full_like(scores_array, target_range[0])
            
            if method == 'minmax':
                # Min-max normalization
                min_val = np.nanmin(scores_array[finite_mask])
                max_val = np.nanmax(scores_array[finite_mask])
                
                if max_val == min_val:
                    # All finite values are the same
                    normalized = np.full_like(scores_array, (target_range[0] + target_range[1]) / 2)
                else:
                    # Standard min-max scaling
                    normalized = (scores_array - min_val) / (max_val - min_val)
                    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
                
            elif method == 'zscore':
                # Z-score normalization
                mean_val = np.nanmean(scores_array[finite_mask])
                std_val = np.nanstd(scores_array[finite_mask])
                
                if std_val == 0:
                    normalized = np.full_like(scores_array, (target_range[0] + target_range[1]) / 2)
                else:
                    normalized = (scores_array - mean_val) / std_val
                    # Scale to target range (assuming 99.7% of data within 3 std devs)
                    normalized = np.clip(normalized, -3, 3)
                    normalized = (normalized + 3) / 6 * (target_range[1] - target_range[0]) + target_range[0]
                
            elif method == 'robust':
                # Robust normalization using median and IQR
                median_val = np.nanmedian(scores_array[finite_mask])
                q75 = np.nanpercentile(scores_array[finite_mask], 75)
                q25 = np.nanpercentile(scores_array[finite_mask], 25)
                iqr = q75 - q25
                
                if iqr == 0:
                    normalized = np.full_like(scores_array, (target_range[0] + target_range[1]) / 2)
                else:
                    normalized = (scores_array - median_val) / iqr
                    # Scale to target range
                    normalized = np.clip(normalized, -2, 2)
                    normalized = (normalized + 2) / 4 * (target_range[1] - target_range[0]) + target_range[0]
            
            else:
                logger.error(f"Unknown normalization method: {method}")
                return scores_array
            
            # Handle infinite and NaN values
            normalized[~finite_mask] = target_range[0]
            
            logger.info(f"Score normalization complete using {method} method")
            return normalized
            
        except Exception as e:
            logger.error(f"Score normalization failed: {e}")
            return np.full_like(np.array(raw_scores), target_range[0])
    
    def calculate_comprehensive_transit_score(self, 
                                            neighborhoods: gpd.GeoDataFrame,
                                            transit_stops: gpd.GeoDataFrame,
                                            service_data: Optional[pd.DataFrame] = None,
                                            distance_method: str = 'euclidean',
                                            **kwargs) -> gpd.GeoDataFrame:
        """
        Calculate comprehensive transit accessibility score combining all factors.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons/points
            transit_stops: GeoDataFrame of transit stop points
            service_data: Optional service frequency data
            distance_method: Method for distance calculation
            **kwargs: Additional parameters
            
        Returns:
            GeoDataFrame with comprehensive transit scores
        """
        try:
            logger.info("Calculating comprehensive transit accessibility scores")
            
            # Step 1: Calculate distances to transit
            result = self.calculate_distance_to_transit(
                neighborhoods, transit_stops, method=distance_method, **kwargs
            )
            
            # Step 2: Weight by service frequency
            result = self.weight_by_frequency(result, transit_stops, service_data)
            
            # Step 3: Adjust for accessibility
            result = self.adjust_for_accessibility(result, transit_stops)
            
            # Step 4: Calculate component scores
            
            # Distance score (inverse - closer is better)
            distance_scores = []
            for _, row in result.iterrows():
                if 'nearest_stop_distance' in row and np.isfinite(row['nearest_stop_distance']):
                    # Convert distance to score (closer = higher score)
                    if row['nearest_stop_distance'] <= self.max_walking_distance:
                        # Linear decay from max distance
                        distance_score = 100 * (1 - row['nearest_stop_distance'] / self.max_walking_distance)
                    else:
                        distance_score = 0.0
                else:
                    distance_score = 0.0
                distance_scores.append(distance_score)
            
            result['distance_score'] = distance_scores
            
            # Coverage score (based on number of stops within range)
            if 'stop_count_within_buffer' in result.columns:
                max_stops = result['stop_count_within_buffer'].max()
                if max_stops > 0:
                    result['coverage_score'] = (result['stop_count_within_buffer'] / max_stops) * 100
                else:
                    result['coverage_score'] = 0.0
            else:
                result['coverage_score'] = 0.0
            
            # Step 5: Combine all scores using weights
            final_scores = (
                self.weights['distance'] * result['distance_score'] +
                self.weights['frequency'] * result['frequency_score'] +
                self.weights['accessibility'] * result['accessibility_score'] +
                self.weights['coverage'] * result['coverage_score']
            )
            
            # Step 6: Normalize final scores
            result['transit_access_score'] = self.normalize_transit_score(final_scores)
            
            # Add score breakdown for analysis
            result['score_breakdown'] = result.apply(
                lambda row: {
                    'distance': row['distance_score'] * self.weights['distance'],
                    'frequency': row['frequency_score'] * self.weights['frequency'],
                    'accessibility': row['accessibility_score'] * self.weights['accessibility'],
                    'coverage': row['coverage_score'] * self.weights['coverage']
                }, axis=1
            )
            
            logger.info("Comprehensive transit score calculation complete")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive transit score calculation failed: {e}")
            # Return original data with default scores
            result = neighborhoods.copy()
            result['transit_access_score'] = 0.0
            return result


# Convenience functions for backward compatibility and easy use

def calculate_distance_to_transit(neighborhoods: gpd.GeoDataFrame,
                                stops: gpd.GeoDataFrame,
                                method: str = 'euclidean',
                                max_distance: float = DEFAULT_MAX_DISTANCE,
                                **kwargs) -> gpd.GeoDataFrame:
    """
    Calculate distances from neighborhoods to transit stops.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhood polygons/centroids
        stops: GeoDataFrame of transit stop points
        method: Distance calculation method
        max_distance: Maximum walking distance to consider
        **kwargs: Additional parameters
        
    Returns:
        GeoDataFrame with distance metrics
    """
    calculator = TransitScoreCalculator(max_walking_distance=max_distance)
    return calculator.calculate_distance_to_transit(neighborhoods, stops, method, **kwargs)


def weight_by_frequency(distances: gpd.GeoDataFrame,
                       schedule: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Weight transit access by service frequency.
    
    Args:
        distances: GeoDataFrame with distance calculations
        schedule: DataFrame with service frequency data
        
    Returns:
        GeoDataFrame with frequency-weighted scores
    """
    calculator = TransitScoreCalculator()
    # Assuming schedule data is properly formatted
    return calculator.weight_by_frequency(distances, None, schedule)


def adjust_for_accessibility(score: gpd.GeoDataFrame,
                           transit_stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Adjust transit scores for wheelchair accessibility.
    
    Args:
        score: GeoDataFrame with transit scores
        transit_stops: GeoDataFrame with accessibility data
        
    Returns:
        GeoDataFrame with accessibility-adjusted scores
    """
    calculator = TransitScoreCalculator()
    return calculator.adjust_for_accessibility(score, transit_stops)


def normalize_transit_score(raw_score: Union[pd.Series, np.ndarray],
                          method: str = 'minmax') -> np.ndarray:
    """
    Normalize transit scores to 0-100 range.
    
    Args:
        raw_score: Raw scores to normalize
        method: Normalization method
        
    Returns:
        Normalized scores array
    """
    calculator = TransitScoreCalculator()
    return calculator.normalize_transit_score(raw_score, method)


def calculate_comprehensive_transit_score(neighborhoods: gpd.GeoDataFrame,
                                        transit_stops: gpd.GeoDataFrame,
                                        service_data: Optional[pd.DataFrame] = None,
                                        **kwargs) -> gpd.GeoDataFrame:
    """
    Calculate comprehensive transit accessibility score.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhoods
        transit_stops: GeoDataFrame of transit stops
        service_data: Optional service frequency data
        **kwargs: Additional parameters
        
    Returns:
        GeoDataFrame with comprehensive transit scores
    """
    calculator = TransitScoreCalculator()
    return calculator.calculate_comprehensive_transit_score(
        neighborhoods, transit_stops, service_data, **kwargs
    )
