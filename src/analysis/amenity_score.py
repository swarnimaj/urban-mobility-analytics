"""
Amenity Proximity Score Development - Sprint 9

This module provides comprehensive amenity proximity scoring for urban mobility analysis.
It evaluates how accessible essential amenities are to each neighborhood, considering
distance, amenity type importance, and accessibility features.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.spatial.distance import cdist
import warnings

# Optional sklearn import
try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Some advanced normalization features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmenityScoreCalculator:
    """
    Calculator for amenity proximity scores based on distance, importance, and accessibility.
    
    This class evaluates how well-served each neighborhood is by essential amenities,
    considering both proximity and accessibility features.
    """
    
    def __init__(self, 
                 max_distance: float = 2000.0,
                 distance_method: str = 'euclidean',
                 weights: Optional[Dict[str, float]] = None,
                 accessibility_penalty: float = 0.5,
                 normalization_method: str = 'minmax'):
        """
        Initialize the AmenityScoreCalculator.
        
        Args:
            max_distance: Maximum distance to consider for amenity access (meters)
            distance_method: Method for calculating distances ('euclidean' or 'network')
            weights: Custom weights for different amenity types
            accessibility_penalty: Penalty factor for inaccessible amenities (0-1)
            normalization_method: Method for normalizing scores ('minmax' or 'zscore')
        """
        self.max_distance = max_distance
        if distance_method not in ['euclidean', 'network']:
            raise ValueError(f"Invalid distance method: {distance_method}. Must be 'euclidean' or 'network'")
        
        self.distance_method = distance_method
        self.accessibility_penalty = accessibility_penalty
        self.normalization_method = normalization_method
        
        # Default amenity type weights (higher = more important)
        self.weights = weights or {
            'hospital': 1.0,           # Critical healthcare
            'clinic': 0.9,             # Healthcare access
            'doctors': 0.9,            # Healthcare access
            'pharmacy': 0.8,           # Healthcare access
            'school': 0.9,             # Education
            'library': 0.7,            # Education/culture
            'community_centre': 0.6,   # Community services
            'bank': 0.6,               # Financial services
            'post_office': 0.5,        # Government services
            'restaurant': 0.4,         # Food services
            'cafe': 0.3,               # Food services
            'bus_station': 0.7,        # Transportation
            'default': 0.5             # Default weight for unknown types
        }
        
        # Accessibility scoring weights
        self.accessibility_weights = {
            'fully_accessible': 1.0,
            'partially_accessible': 0.7,
            'not_accessible': 0.3,
            'unknown': 0.5
        }
        
        logger.info(f"Initialized AmenityScoreCalculator with max_distance={max_distance}m, "
                   f"method={distance_method}, penalty={accessibility_penalty}")
    
    def calculate_amenity_distances(self, 
                                  neighborhoods: gpd.GeoDataFrame, 
                                  amenities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate distances from each neighborhood to all amenities.
        
        Args:
            neighborhoods: GeoDataFrame with neighborhood geometries
            amenities: GeoDataFrame with amenity locations and types
            
        Returns:
            GeoDataFrame with distance calculations for each neighborhood-amenity pair
        """
        logger.info(f"Calculating amenity distances for {len(neighborhoods)} neighborhoods "
                   f"and {len(amenities)} amenities using {self.distance_method} method")
        
        try:
            # Ensure both datasets are in the same CRS
            if neighborhoods.crs != amenities.crs:
                amenities = amenities.to_crs(neighborhoods.crs)
                logger.info(f"Converted amenities to CRS: {neighborhoods.crs}")
            
            # Convert to projected CRS for accurate distance calculations
            neighborhoods_proj = neighborhoods.to_crs('EPSG:3857')
            amenities_proj = amenities.to_crs('EPSG:3857')
            
            # Get neighborhood centroids
            neighborhood_centroids = neighborhoods_proj.geometry.centroid
            
            # Get amenity points
            amenity_points = amenities_proj.geometry
            
            # Calculate distances
            if self.distance_method == 'euclidean':
                distances = self._calculate_euclidean_distances(neighborhood_centroids, amenity_points)
            elif self.distance_method == 'network':
                distances = self._calculate_network_distances(neighborhood_centroids, amenity_points)
            else:
                raise ValueError(f"Unknown distance method: {self.distance_method}")
            
            # Create results DataFrame
            results = []
            for i, (idx, neighborhood) in enumerate(neighborhoods.iterrows()):
                for j, (amenity_idx, amenity) in enumerate(amenities.iterrows()):
                    distance = distances[i, j]
                    
                    # Only include amenities within max_distance
                    if distance <= self.max_distance:
                        results.append({
                            'neighborhood_id': idx,
                            'amenity_id': amenity_idx,
                            'amenity_type': amenity.get('amenity', 'unknown'),
                            'amenity_name': amenity.get('name', 'Unknown'),
                            'distance_m': distance,
                            'accessibility_score': amenity.get('accessibility_score', 0),
                            'wheelchair_score': amenity.get('wheelchair_score', 0),
                            'accessibility_category': amenity.get('accessibility_category', 'unknown')
                        })
            
            result_df = pd.DataFrame(results)
            logger.info(f"Calculated distances for {len(result_df)} neighborhood-amenity pairs")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating amenity distances: {e}")
            raise
    
    def _calculate_euclidean_distances(self, 
                                     neighborhood_centroids: gpd.GeoSeries, 
                                     amenity_points: gpd.GeoSeries) -> np.ndarray:
        """Calculate Euclidean distances between neighborhood centroids and amenities."""
        # Handle empty amenity points
        if len(amenity_points) == 0:
            return np.array([]).reshape(len(neighborhood_centroids), 0)
        
        # Extract coordinates - handle both Point and MultiPolygon geometries
        neighborhood_coords = []
        for geom in neighborhood_centroids:
            if hasattr(geom, 'x') and hasattr(geom, 'y'):  # Point geometry
                neighborhood_coords.append([geom.x, geom.y])
            else:  # Polygon geometry - use centroid
                centroid = geom.centroid
                neighborhood_coords.append([centroid.x, centroid.y])
        
        amenity_coords = []
        for geom in amenity_points:
            if hasattr(geom, 'x') and hasattr(geom, 'y'):  # Point geometry
                amenity_coords.append([geom.x, geom.y])
            else:  # Polygon geometry - use centroid
                centroid = geom.centroid
                amenity_coords.append([centroid.x, centroid.y])
        
        # Convert to numpy arrays
        neighborhood_coords = np.array(neighborhood_coords)
        amenity_coords = np.array(amenity_coords)
        
        # Calculate distances using scipy
        distances = cdist(neighborhood_coords, amenity_coords, metric='euclidean')
        
        return distances
    
    def _calculate_network_distances(self, 
                                   neighborhood_centroids: gpd.GeoSeries, 
                                   amenity_points: gpd.GeoSeries) -> np.ndarray:
        """
        Calculate network distances (placeholder for future network analysis).
        For now, falls back to Euclidean distances.
        """
        logger.warning("Network distance calculation not yet implemented, using Euclidean distances")
        return self._calculate_euclidean_distances(neighborhood_centroids, amenity_points)
    
    def weight_by_amenity_type(self, distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply importance weights based on amenity type.
        
        Args:
            distances_df: DataFrame with distance calculations
            
        Returns:
            DataFrame with weighted scores
        """
        logger.info("Applying amenity type weights")
        
        try:
            # Create a copy to avoid modifying original
            weighted_df = distances_df.copy()
            
            # Apply amenity type weights
            weighted_df['amenity_weight'] = weighted_df['amenity_type'].map(self.weights).fillna(self.weights['default'])
            
            # Calculate weighted distance score (inverse of distance, weighted by importance)
            # Closer amenities get higher scores
            weighted_df['distance_score'] = (1.0 / (weighted_df['distance_m'] + 1)) * weighted_df['amenity_weight']
            
            # Apply accessibility weights
            weighted_df['accessibility_weight'] = weighted_df['accessibility_category'].map(self.accessibility_weights).fillna(0.5)
            
            # Calculate final weighted score
            weighted_df['weighted_score'] = (
                weighted_df['distance_score'] * 
                weighted_df['accessibility_weight'] * 
                (1.0 - self.accessibility_penalty * (1.0 - weighted_df['accessibility_weight']))
            )
            
            logger.info(f"Applied weights to {len(weighted_df)} amenity-neighborhood pairs")
            
            return weighted_df
            
        except Exception as e:
            logger.error(f"Error applying amenity type weights: {e}")
            raise
    
    def calculate_accessibility_penalty(self, 
                                      weighted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply accessibility penalties for inaccessible amenities.
        
        Args:
            weighted_df: DataFrame with weighted scores
            
        Returns:
            DataFrame with accessibility penalties applied
        """
        logger.info("Calculating accessibility penalties")
        
        try:
            penalty_df = weighted_df.copy()
            
            # Define accessibility penalty factors
            penalty_factors = {
                'not_accessible': self.accessibility_penalty,
                'partially_accessible': self.accessibility_penalty * 0.5,
                'fully_accessible': 0.0,
                'unknown': self.accessibility_penalty * 0.3
            }
            
            # Apply penalties
            penalty_df['accessibility_penalty'] = penalty_df['accessibility_category'].map(penalty_factors).fillna(self.accessibility_penalty * 0.3)
            penalty_df['final_score'] = penalty_df['weighted_score'] * (1.0 - penalty_df['accessibility_penalty'])
            
            # Calculate accessibility-adjusted distance (penalized distance) if distance_m exists
            if 'distance_m' in penalty_df.columns:
                penalty_df['penalized_distance'] = penalty_df['distance_m'] * (1.0 + penalty_df['accessibility_penalty'])
            else:
                penalty_df['penalized_distance'] = 0.0
            
            logger.info(f"Applied accessibility penalties to {len(penalty_df)} amenity-neighborhood pairs")
            
            return penalty_df
            
        except Exception as e:
            logger.error(f"Error calculating accessibility penalties: {e}")
            raise
    
    def normalize_amenity_score(self, 
                              raw_scores: Union[pd.Series, np.ndarray], 
                              method: Optional[str] = None) -> np.ndarray:
        """
        Normalize amenity scores to 0-100 range.
        
        Args:
            raw_scores: Raw amenity scores to normalize
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Normalized scores in 0-100 range
        """
        if method is None:
            method = self.normalization_method
            
        logger.info(f"Normalizing amenity scores using {method} method")
        
        try:
            # Convert to numpy array if needed
            if isinstance(raw_scores, pd.Series):
                scores_array = raw_scores.values
            else:
                scores_array = raw_scores
            
            # Remove NaN values for calculation
            valid_scores = scores_array[~np.isnan(scores_array)]
            
            if len(valid_scores) == 0:
                logger.warning("No valid scores found for normalization")
                return np.zeros_like(scores_array)
            
            if method == 'minmax':
                # Min-Max normalization to 0-100
                min_score = np.min(valid_scores)
                max_score = np.max(valid_scores)
                
                if max_score == min_score:
                    # All scores are the same
                    normalized = np.full_like(scores_array, 50.0)
                else:
                    normalized = ((scores_array - min_score) / (max_score - min_score)) * 100
                    
            elif method == 'zscore':
                # Z-score normalization with clipping to 0-100
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                
                if std_score == 0:
                    # All scores are the same
                    normalized = np.full_like(scores_array, 50.0)
                else:
                    z_scores = (scores_array - mean_score) / std_score
                    # Convert z-scores to 0-100 range (assuming normal distribution)
                    normalized = np.clip(50 + (z_scores * 20), 0, 100)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Ensure scores are in valid range
            normalized = np.clip(normalized, 0, 100)
            
            logger.info(f"Normalized {len(scores_array)} scores to range [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing amenity scores: {e}")
            raise
    
    def calculate_comprehensive_amenity_score(self, 
                                            neighborhoods: gpd.GeoDataFrame, 
                                            amenities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate comprehensive amenity proximity scores for all neighborhoods.
        
        Args:
            neighborhoods: GeoDataFrame with neighborhood geometries
            amenities: GeoDataFrame with amenity locations and types
            
        Returns:
            GeoDataFrame with amenity scores added to neighborhoods
        """
        logger.info(f"Calculating comprehensive amenity scores for {len(neighborhoods)} neighborhoods")
        
        try:
            # Step 1: Calculate distances
            distances_df = self.calculate_amenity_distances(neighborhoods, amenities)
            
            if distances_df.empty:
                logger.warning("No amenities found within max_distance, returning zero scores")
                result = neighborhoods.copy()
                result['amenity_access_score'] = 0.0
                result['amenity_density'] = 0.0
                result['amenity_diversity'] = 0.0
                result['accessibility_score'] = 0.0
                return result
            
            # Step 2: Apply amenity type weights
            weighted_df = self.weight_by_amenity_type(distances_df)
            
            # Step 3: Apply accessibility penalties
            penalty_df = self.calculate_accessibility_penalty(weighted_df)
            
            # Step 4: Aggregate scores by neighborhood
            neighborhood_scores = self._aggregate_neighborhood_scores(penalty_df, neighborhoods)
            
            # Step 5: Normalize final scores
            neighborhood_scores['amenity_access_score'] = self.normalize_amenity_score(
                neighborhood_scores['raw_amenity_score']
            )
            
            # Merge with original neighborhoods
            result = neighborhoods.merge(
                neighborhood_scores[['neighborhood_id', 'amenity_access_score', 'amenity_density', 
                                   'amenity_diversity', 'accessibility_score']], 
                left_index=True, 
                right_on='neighborhood_id', 
                how='left'
            ).fillna(0.0)
            
            logger.info(f"Calculated comprehensive amenity scores for {len(result)} neighborhoods")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive amenity scores: {e}")
            raise
    
    def _aggregate_neighborhood_scores(self, 
                                     penalty_df: pd.DataFrame, 
                                     neighborhoods: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate amenity scores by neighborhood.
        
        Args:
            penalty_df: DataFrame with penalized scores
            neighborhoods: Original neighborhoods GeoDataFrame
            
        Returns:
            DataFrame with aggregated scores per neighborhood
        """
        logger.info("Aggregating neighborhood amenity scores")
        
        try:
            # Group by neighborhood and calculate aggregated metrics
            agg_scores = penalty_df.groupby('neighborhood_id').agg({
                'final_score': ['sum', 'mean', 'count'],
                'amenity_type': 'nunique',
                'accessibility_weight': 'mean',
                'distance_m': 'mean'
            }).round(4)
            
            # Flatten column names
            agg_scores.columns = ['raw_amenity_score', 'avg_amenity_score', 'amenity_count', 
                                'amenity_diversity', 'avg_accessibility', 'avg_distance']
            
            # Calculate additional metrics
            agg_scores['amenity_density'] = agg_scores['amenity_count'] / len(neighborhoods)
            agg_scores['accessibility_score'] = agg_scores['avg_accessibility'] * 100
            
            # Reset index
            agg_scores = agg_scores.reset_index()
            
            logger.info(f"Aggregated scores for {len(agg_scores)} neighborhoods")
            
            return agg_scores
            
        except Exception as e:
            logger.error(f"Error aggregating neighborhood scores: {e}")
            raise
    
    def validate_amenity_data(self, 
                            amenities: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate amenity data quality and completeness.
        
        Args:
            amenities: GeoDataFrame with amenity data
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating amenity data quality")
        
        try:
            # Handle empty amenities
            if len(amenities) == 0:
                return {
                    'total_amenities': 0,
                    'amenity_types': 0,
                    'has_geometry': False,
                    'has_accessibility': False,
                    'has_wheelchair': False,
                    'valid_geometries': 0,
                    'missing_amenity_types': 0,
                    'accessibility_coverage': 0.0,
                    'overall_quality_score': 0.0
                }
            
            validation_results = {
                'total_amenities': len(amenities),
                'amenity_types': amenities['amenity'].nunique() if 'amenity' in amenities.columns else 0,
                'has_geometry': 'geometry' in amenities.columns,
                'has_accessibility': 'accessibility_score' in amenities.columns,
                'has_wheelchair': 'wheelchair_score' in amenities.columns,
                'valid_geometries': amenities.geometry.is_valid.sum() if 'geometry' in amenities.columns else 0,
                'missing_amenity_types': amenities['amenity'].isna().sum() if 'amenity' in amenities.columns else 0,
                'accessibility_coverage': 0.0,
                'overall_quality_score': 0.0
            }
            
            # Calculate accessibility coverage
            if 'accessibility_category' in amenities.columns:
                accessible_count = len(amenities[amenities['accessibility_category'] != 'unknown'])
                validation_results['accessibility_coverage'] = accessible_count / len(amenities)
            
            # Calculate overall quality score
            quality_factors = [
                validation_results['has_geometry'],
                validation_results['has_accessibility'],
                validation_results['valid_geometries'] / max(1, len(amenities)),
                1.0 - (validation_results['missing_amenity_types'] / max(1, len(amenities))),
                validation_results['accessibility_coverage']
            ]
            
            validation_results['overall_quality_score'] = np.mean(quality_factors)
            
            logger.info(f"Amenity data validation complete. Quality score: {validation_results['overall_quality_score']:.3f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating amenity data: {e}")
            return {'error': str(e), 'overall_quality_score': 0.0}


# Convenience functions for direct use
def calculate_amenity_distances(neighborhoods: gpd.GeoDataFrame, 
                              amenities: gpd.GeoDataFrame,
                              max_distance: float = 2000.0,
                              distance_method: str = 'euclidean') -> pd.DataFrame:
    """
    Calculate distances from neighborhoods to amenities.
    
    Args:
        neighborhoods: GeoDataFrame with neighborhood geometries
        amenities: GeoDataFrame with amenity locations
        max_distance: Maximum distance to consider (meters)
        distance_method: Method for calculating distances
        
    Returns:
        DataFrame with distance calculations
    """
    calculator = AmenityScoreCalculator(max_distance=max_distance, distance_method=distance_method)
    return calculator.calculate_amenity_distances(neighborhoods, amenities)


def weight_by_amenity_type(distances_df: pd.DataFrame, 
                          weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Apply importance weights to amenity distances.
    
    Args:
        distances_df: DataFrame with distance calculations
        weights: Custom weights for amenity types
        
    Returns:
        DataFrame with weighted scores
    """
    calculator = AmenityScoreCalculator(weights=weights)
    return calculator.weight_by_amenity_type(distances_df)


def calculate_accessibility_penalty(weighted_df: pd.DataFrame,
                                  penalty_factor: float = 0.5) -> pd.DataFrame:
    """
    Apply accessibility penalties to weighted scores.
    
    Args:
        weighted_df: DataFrame with weighted scores
        penalty_factor: Penalty factor for inaccessible amenities
        
    Returns:
        DataFrame with accessibility penalties applied
    """
    calculator = AmenityScoreCalculator(accessibility_penalty=penalty_factor)
    return calculator.calculate_accessibility_penalty(weighted_df)


def normalize_amenity_score(raw_scores: Union[pd.Series, np.ndarray],
                          method: str = 'minmax') -> np.ndarray:
    """
    Normalize amenity scores to 0-100 range.
    
    Args:
        raw_scores: Raw scores to normalize
        method: Normalization method
        
    Returns:
        Normalized scores
    """
    calculator = AmenityScoreCalculator(normalization_method=method)
    return calculator.normalize_amenity_score(raw_scores)
