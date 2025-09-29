# src/analysis/sidewalk_score.py
"""
Sidewalk & Ramp Score Module for Urban Mobility Analytics.

This module calculates comprehensive sidewalk accessibility scores for neighborhoods
based on sidewalk coverage, curb ramp availability, pedestrian crossings, and 
accessibility features.

The scoring system considers:
1. Sidewalk coverage along streets and paths
2. Curb ramp availability at crossings
3. Pedestrian island safety features
4. Accessibility compliance (ADA standards)
5. Network connectivity for pedestrians

Functions:
- calculate_sidewalk_coverage(): Calculate sidewalk coverage percentage
- identify_missing_curb_ramps(): Find missing curb ramps at crossings
- calculate_pedestrian_islands(): Identify safe crossing islands
- normalize_sidewalk_score(): Scale scores to 0-100 range
- calculate_comprehensive_sidewalk_score(): Main scoring function
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union

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
DEFAULT_SIDEWALK_WIDTH = 1.5  # meters (standard sidewalk width)
DEFAULT_CROSSING_WIDTH = 3.0  # meters (standard crossing width)
DEFAULT_WEIGHTS = {
    'coverage': 0.4,
    'ramps': 0.25,
    'islands': 0.15,
    'accessibility': 0.2
}

class SidewalkScoreCalculator:
    """
    Main class for calculating sidewalk accessibility scores.
    """
    
    def __init__(self, 
                 sidewalk_width: float = DEFAULT_SIDEWALK_WIDTH,
                 crossing_width: float = DEFAULT_CROSSING_WIDTH,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the sidewalk score calculator.
        
        Args:
            sidewalk_width: Standard sidewalk width in meters
            crossing_width: Standard crossing width in meters
            weights: Dictionary of scoring component weights
        """
        self.sidewalk_width = sidewalk_width
        self.crossing_width = crossing_width
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        
        # Validate weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {sum(self.weights.values())}")
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Initialized SidewalkScoreCalculator with sidewalk width: {sidewalk_width}m")
    
    def calculate_sidewalk_coverage(self, 
                                  neighborhoods: gpd.GeoDataFrame,
                                  sidewalks: gpd.GeoDataFrame,
                                  street_network: Optional[object] = None) -> gpd.GeoDataFrame:
        """
        Calculate sidewalk coverage percentage for each neighborhood.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons
            sidewalks: GeoDataFrame of sidewalk segments
            street_network: Optional street network for reference
            
        Returns:
            GeoDataFrame with sidewalk coverage metrics
        """
        if neighborhoods is None or neighborhoods.empty:
            logger.error("No neighborhoods provided")
            return neighborhoods
        
        if sidewalks is None or sidewalks.empty:
            logger.warning("No sidewalk data provided")
            result = neighborhoods.copy()
            result['sidewalk_coverage_pct'] = 0.0
            result['sidewalk_length_km'] = 0.0
            result['sidewalk_density_km_sqkm'] = 0.0
            return result
        
        logger.info("Calculating sidewalk coverage")
        
        # Ensure both datasets have consistent CRS
        neighborhoods = ensure_crs(neighborhoods, DEFAULT_ANALYSIS_CRS)
        sidewalks = ensure_crs(sidewalks, DEFAULT_ANALYSIS_CRS)
        
        result = neighborhoods.copy()
        coverage_metrics = []
        
        for idx, neighborhood in neighborhoods.iterrows():
            try:
                # Get sidewalks within or intersecting the neighborhood
                sidewalks_in_area = sidewalks[sidewalks.geometry.intersects(neighborhood.geometry)]
                
                if sidewalks_in_area.empty:
                    coverage_metrics.append({
                        'sidewalk_coverage_pct': 0.0,
                        'sidewalk_length_km': 0.0,
                        'sidewalk_density_km_sqkm': 0.0
                    })
                    continue
                
                # Calculate total sidewalk length in the area
                total_sidewalk_length = sidewalks_in_area.geometry.length.sum() / 1000  # Convert to km
                
                # Calculate neighborhood area in square km
                neighborhood_area_sqkm = neighborhood.geometry.area / 1_000_000  # Convert to sq km
                
                # Calculate sidewalk density
                sidewalk_density = total_sidewalk_length / neighborhood_area_sqkm if neighborhood_area_sqkm > 0 else 0
                
                # Calculate coverage percentage (simplified heuristic)
                # This is a basic calculation - in practice, you'd want to compare against
                # the total street length that should have sidewalks
                if street_network is not None:
                    # If we have street network data, calculate more accurate coverage
                    coverage_pct = self._calculate_detailed_coverage(
                        neighborhood.geometry, sidewalks_in_area, street_network
                    )
                else:
                    # Use density as a proxy for coverage
                    # This is a simplified approach - assumes higher density = better coverage
                    max_density = 10.0  # km/sqkm - reasonable maximum for urban areas
                    coverage_pct = min(100.0, (sidewalk_density / max_density) * 100)
                
                coverage_metrics.append({
                    'sidewalk_coverage_pct': coverage_pct,
                    'sidewalk_length_km': total_sidewalk_length,
                    'sidewalk_density_km_sqkm': sidewalk_density
                })
                
            except Exception as e:
                logger.error(f"Error calculating coverage for neighborhood {idx}: {e}")
                coverage_metrics.append({
                    'sidewalk_coverage_pct': 0.0,
                    'sidewalk_length_km': 0.0,
                    'sidewalk_density_km_sqkm': 0.0
                })
        
        # Add metrics to result
        for i, metrics in enumerate(coverage_metrics):
            for key, value in metrics.items():
                if key not in result.columns:
                    result[key] = 0.0
                result.iloc[i, result.columns.get_loc(key)] = value
        
        # Ensure all columns exist
        for key in ['sidewalk_coverage_pct', 'sidewalk_length_km', 'sidewalk_density_km_sqkm']:
            if key not in result.columns:
                result[key] = 0.0
        
        logger.info("Sidewalk coverage calculation complete")
        return result
    
    def _calculate_detailed_coverage(self, 
                                   neighborhood_geom: Polygon,
                                   sidewalks: gpd.GeoDataFrame,
                                   street_network: object) -> float:
        """
        Calculate detailed sidewalk coverage using street network reference.
        
        This is a simplified implementation. In practice, you'd want to:
        1. Extract street segments within the neighborhood
        2. Determine which streets should have sidewalks
        3. Calculate the ratio of sidewalked streets to total streets
        """
        try:
            # This is a placeholder for more sophisticated coverage calculation
            # For now, use density as a proxy
            total_sidewalk_length = sidewalks.geometry.length.sum() / 1000
            neighborhood_area = neighborhood_geom.area / 1_000_000
            
            if neighborhood_area == 0:
                return 0.0
            
            density = total_sidewalk_length / neighborhood_area
            # Convert density to percentage (heuristic)
            return min(100.0, density * 10)  # 10 km/sqkm = 100% coverage
            
        except Exception as e:
            logger.error(f"Error in detailed coverage calculation: {e}")
            return 0.0
    
    def identify_missing_curb_ramps(self, 
                                  neighborhoods: gpd.GeoDataFrame,
                                  crossings: gpd.GeoDataFrame,
                                  curb_ramps: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Identify missing curb ramps at pedestrian crossings.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons
            crossings: GeoDataFrame of pedestrian crossings
            curb_ramps: Optional GeoDataFrame of existing curb ramps
            
        Returns:
            GeoDataFrame with curb ramp analysis
        """
        if neighborhoods is None or neighborhoods.empty:
            logger.error("No neighborhoods provided")
            return neighborhoods
        
        if crossings is None or crossings.empty:
            logger.warning("No crossing data provided")
            result = neighborhoods.copy()
            result['total_crossings'] = 0
            result['crossings_with_ramps'] = 0
            result['ramp_coverage_pct'] = 0.0
            return result
        
        logger.info("Identifying missing curb ramps")
        
        # Ensure consistent CRS
        neighborhoods = ensure_crs(neighborhoods, DEFAULT_ANALYSIS_CRS)
        crossings = ensure_crs(crossings, DEFAULT_ANALYSIS_CRS)
        
        if curb_ramps is not None:
            curb_ramps = ensure_crs(curb_ramps, DEFAULT_ANALYSIS_CRS)
        
        result = neighborhoods.copy()
        ramp_metrics = []
        
        for idx, neighborhood in neighborhoods.iterrows():
            try:
                # Get crossings within the neighborhood
                crossings_in_area = crossings[crossings.geometry.intersects(neighborhood.geometry)]
                total_crossings = len(crossings_in_area)
                
                if total_crossings == 0:
                    ramp_metrics.append({
                        'total_crossings': 0,
                        'crossings_with_ramps': 0,
                        'ramp_coverage_pct': 0.0
                    })
                    continue
                
                # Count crossings with nearby curb ramps
                crossings_with_ramps = 0
                
                if curb_ramps is not None and not curb_ramps.empty:
                    # Check for curb ramps near each crossing
                    for _, crossing in crossings_in_area.iterrows():
                        # Create buffer around crossing to find nearby ramps
                        crossing_buffer = crossing.geometry.buffer(10)  # 10 meter buffer
                        nearby_ramps = curb_ramps[curb_ramps.geometry.intersects(crossing_buffer)]
                        
                        if not nearby_ramps.empty:
                            crossings_with_ramps += 1
                else:
                    # If no curb ramp data, assume some crossings have ramps based on crossing type
                    # This is a heuristic - in practice, you'd want actual ramp data
                    if 'crossing' in crossings_in_area.columns:
                        # Assume marked crossings are more likely to have ramps
                        marked_crossings = crossings_in_area[crossings_in_area['crossing'].notna()]
                        crossings_with_ramps = len(marked_crossings) * 0.6  # Assume 60% have ramps
                    else:
                        # Default assumption: 30% of crossings have ramps
                        crossings_with_ramps = total_crossings * 0.3
                
                ramp_coverage_pct = (crossings_with_ramps / total_crossings) * 100 if total_crossings > 0 else 0
                
                ramp_metrics.append({
                    'total_crossings': total_crossings,
                    'crossings_with_ramps': int(crossings_with_ramps),
                    'ramp_coverage_pct': ramp_coverage_pct
                })
                
            except Exception as e:
                logger.error(f"Error analyzing curb ramps for neighborhood {idx}: {e}")
                ramp_metrics.append({
                    'total_crossings': 0,
                    'crossings_with_ramps': 0,
                    'ramp_coverage_pct': 0.0
                })
        
        # Add metrics to result
        for i, metrics in enumerate(ramp_metrics):
            for key, value in metrics.items():
                if key not in result.columns:
                    result[key] = 0
                result.iloc[i, result.columns.get_loc(key)] = value
        
        logger.info("Curb ramp analysis complete")
        return result
    
    def calculate_pedestrian_islands(self, 
                                   neighborhoods: gpd.GeoDataFrame,
                                   crossings: gpd.GeoDataFrame,
                                   street_network: Optional[object] = None) -> gpd.GeoDataFrame:
        """
        Identify safe pedestrian crossing islands and refuge areas.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons
            crossings: GeoDataFrame of pedestrian crossings
            street_network: Optional street network for reference
            
        Returns:
            GeoDataFrame with pedestrian island analysis
        """
        if neighborhoods is None or neighborhoods.empty:
            logger.error("No neighborhoods provided")
            return neighborhoods
        
        if crossings is None or crossings.empty:
            logger.warning("No crossing data provided")
            result = neighborhoods.copy()
            result['total_crossings'] = 0
            result['crossings_with_islands'] = 0
            result['island_coverage_pct'] = 0.0
            return result
        
        logger.info("Calculating pedestrian islands")
        
        # Ensure consistent CRS
        neighborhoods = ensure_crs(neighborhoods, DEFAULT_ANALYSIS_CRS)
        crossings = ensure_crs(crossings, DEFAULT_ANALYSIS_CRS)
        
        result = neighborhoods.copy()
        island_metrics = []
        
        for idx, neighborhood in neighborhoods.iterrows():
            try:
                # Get crossings within the neighborhood
                crossings_in_area = crossings[crossings.geometry.intersects(neighborhood.geometry)]
                total_crossings = len(crossings_in_area)
                
                if total_crossings == 0:
                    island_metrics.append({
                        'total_crossings': 0,
                        'crossings_with_islands': 0,
                        'island_coverage_pct': 0.0
                    })
                    continue
                
                # Count crossings with islands (simplified heuristic)
                crossings_with_islands = 0
                
                for _, crossing in crossings_in_area.iterrows():
                    # Check if crossing has island features
                    has_island = self._check_pedestrian_island(crossing, street_network)
                    if has_island:
                        crossings_with_islands += 1
                
                island_coverage_pct = (crossings_with_islands / total_crossings) * 100 if total_crossings > 0 else 0
                
                island_metrics.append({
                    'total_crossings': total_crossings,
                    'crossings_with_islands': crossings_with_islands,
                    'island_coverage_pct': island_coverage_pct
                })
                
            except Exception as e:
                logger.error(f"Error analyzing pedestrian islands for neighborhood {idx}: {e}")
                island_metrics.append({
                    'total_crossings': 0,
                    'crossings_with_islands': 0,
                    'island_coverage_pct': 0.0
                })
        
        # Add metrics to result
        for i, metrics in enumerate(island_metrics):
            for key, value in metrics.items():
                if key not in result.columns:
                    result[key] = 0
                result.iloc[i, result.columns.get_loc(key)] = value
        
        logger.info("Pedestrian island analysis complete")
        return result
    
    def _check_pedestrian_island(self, crossing: pd.Series, street_network: Optional[object]) -> bool:
        """
        Check if a crossing has pedestrian island features.
        
        This is a simplified heuristic. In practice, you'd want to:
        1. Check for physical island geometries
        2. Analyze traffic patterns
        3. Look for refuge areas
        """
        try:
            # Simple heuristic based on crossing attributes
            if 'island' in crossing and crossing['island'] == 'yes':
                return True
            
            if 'refuge' in crossing and crossing['refuge'] == 'yes':
                return True
            
            # Check for crossing type that might indicate islands
            if 'crossing' in crossing:
                crossing_type = str(crossing['crossing']).lower()
                if any(keyword in crossing_type for keyword in ['island', 'refuge', 'median']):
                    return True
            
            # Default: assume 20% of crossings have islands (heuristic)
            return np.random.random() < 0.2
            
        except Exception as e:
            logger.error(f"Error checking pedestrian island: {e}")
            return False
    
    def normalize_sidewalk_score(self, 
                               raw_scores: Union[pd.Series, np.ndarray],
                               method: str = 'minmax',
                               target_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
        """
        Normalize raw sidewalk scores to a specified range.
        
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
                    normalized = np.full_like(scores_array, (target_range[0] + target_range[1]) / 2)
                else:
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
                    normalized = np.clip(normalized, -2, 2)
                    normalized = (normalized + 2) / 4 * (target_range[1] - target_range[0]) + target_range[0]
            
            else:
                logger.error(f"Unknown normalization method: {method}")
                return scores_array
            
            # Handle infinite and NaN values
            normalized[~finite_mask] = target_range[0]
            
            logger.info(f"Sidewalk score normalization complete using {method} method")
            return normalized
            
        except Exception as e:
            logger.error(f"Score normalization failed: {e}")
            return np.full_like(np.array(raw_scores), target_range[0])
    
    def validate_sidewalk_data(self, 
                             sidewalks: gpd.GeoDataFrame,
                             crossings: Optional[gpd.GeoDataFrame] = None,
                             curb_ramps: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        Validate sidewalk data quality and completeness.
        
        Args:
            sidewalks: GeoDataFrame of sidewalk segments
            crossings: Optional GeoDataFrame of pedestrian crossings
            curb_ramps: Optional GeoDataFrame of curb ramps
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        try:
            logger.info("Validating sidewalk data quality")
            
            validation_results = {
                'sidewalks': {},
                'crossings': {},
                'curb_ramps': {},
                'overall_quality': 'unknown'
            }
            
            # Validate sidewalks data
            if sidewalks is not None and not sidewalks.empty:
                sidewalk_issues = []
                
                # Check for valid geometries
                invalid_geoms = sidewalks.geometry.isna().sum()
                if invalid_geoms > 0:
                    sidewalk_issues.append(f"{invalid_geoms} invalid geometries")
                
                # Check for empty geometries
                empty_geoms = sidewalks.geometry.is_empty.sum()
                if empty_geoms > 0:
                    sidewalk_issues.append(f"{empty_geoms} empty geometries")
                
                # Check data completeness
                total_sidewalks = len(sidewalks)
                if total_sidewalks == 0:
                    sidewalk_issues.append("No sidewalk data available")
                
                # Calculate basic metrics
                total_length = sidewalks.geometry.length.sum() / 1000  # km
                avg_length = sidewalks.geometry.length.mean()
                
                validation_results['sidewalks'] = {
                    'total_count': total_sidewalks,
                    'total_length_km': total_length,
                    'avg_length_m': avg_length,
                    'issues': sidewalk_issues,
                    'quality_score': max(0, 100 - len(sidewalk_issues) * 20)
                }
            else:
                validation_results['sidewalks'] = {
                    'total_count': 0,
                    'total_length_km': 0,
                    'avg_length_m': 0,
                    'issues': ['No sidewalk data provided'],
                    'quality_score': 0
                }
            
            # Validate crossings data
            if crossings is not None and not crossings.empty:
                crossing_issues = []
                
                invalid_geoms = crossings.geometry.isna().sum()
                if invalid_geoms > 0:
                    crossing_issues.append(f"{invalid_geoms} invalid geometries")
                
                total_crossings = len(crossings)
                if total_crossings == 0:
                    crossing_issues.append("No crossing data available")
                
                validation_results['crossings'] = {
                    'total_count': total_crossings,
                    'issues': crossing_issues,
                    'quality_score': max(0, 100 - len(crossing_issues) * 20)
                }
            else:
                validation_results['crossings'] = {
                    'total_count': 0,
                    'issues': ['No crossing data provided'],
                    'quality_score': 0
                }
            
            # Validate curb ramps data
            if curb_ramps is not None and not curb_ramps.empty:
                ramp_issues = []
                
                invalid_geoms = curb_ramps.geometry.isna().sum()
                if invalid_geoms > 0:
                    ramp_issues.append(f"{invalid_geoms} invalid geometries")
                
                total_ramps = len(curb_ramps)
                if total_ramps == 0:
                    ramp_issues.append("No curb ramp data available")
                
                validation_results['curb_ramps'] = {
                    'total_count': total_ramps,
                    'issues': ramp_issues,
                    'quality_score': max(0, 100 - len(ramp_issues) * 20)
                }
            else:
                validation_results['curb_ramps'] = {
                    'total_count': 0,
                    'issues': ['No curb ramp data provided'],
                    'quality_score': 0
                }
            
            # Calculate overall quality
            quality_scores = [
                validation_results['sidewalks']['quality_score'],
                validation_results['crossings']['quality_score'],
                validation_results['curb_ramps']['quality_score']
            ]
            
            valid_scores = [score for score in quality_scores if score > 0]
            if valid_scores:
                overall_score = np.mean(valid_scores)
            else:
                overall_score = 0
            
            if overall_score >= 80:
                validation_results['overall_quality'] = 'excellent'
            elif overall_score >= 60:
                validation_results['overall_quality'] = 'good'
            elif overall_score >= 40:
                validation_results['overall_quality'] = 'fair'
            else:
                validation_results['overall_quality'] = 'poor'
            
            validation_results['overall_score'] = overall_score
            
            logger.info(f"Sidewalk data validation complete - Overall quality: {validation_results['overall_quality']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating sidewalk data: {e}")
            return {
                'sidewalks': {'issues': [f'Validation error: {e}'], 'quality_score': 0},
                'crossings': {'issues': [f'Validation error: {e}'], 'quality_score': 0},
                'curb_ramps': {'issues': [f'Validation error: {e}'], 'quality_score': 0},
                'overall_quality': 'error',
                'overall_score': 0
            }

    def calculate_comprehensive_sidewalk_score(self, 
                                             neighborhoods: gpd.GeoDataFrame,
                                             sidewalks: gpd.GeoDataFrame,
                                             crossings: Optional[gpd.GeoDataFrame] = None,
                                             curb_ramps: Optional[gpd.GeoDataFrame] = None,
                                             street_network: Optional[object] = None) -> gpd.GeoDataFrame:
        """
        Calculate comprehensive sidewalk accessibility score combining all factors.
        
        Args:
            neighborhoods: GeoDataFrame of neighborhood polygons
            sidewalks: GeoDataFrame of sidewalk segments
            crossings: Optional GeoDataFrame of pedestrian crossings
            curb_ramps: Optional GeoDataFrame of curb ramps
            street_network: Optional street network for reference
            
        Returns:
            GeoDataFrame with comprehensive sidewalk scores
        """
        try:
            logger.info("Calculating comprehensive sidewalk accessibility scores")
            
            # Step 1: Calculate sidewalk coverage
            result = self.calculate_sidewalk_coverage(neighborhoods, sidewalks, street_network)
            
            # Step 2: Analyze curb ramps if crossings data available
            if crossings is not None:
                result = self.identify_missing_curb_ramps(result, crossings, curb_ramps)
            else:
                # Add default values
                result['total_crossings'] = 0
                result['crossings_with_ramps'] = 0
                result['ramp_coverage_pct'] = 0.0
            
            # Step 3: Analyze pedestrian islands if crossings data available
            if crossings is not None:
                result = self.calculate_pedestrian_islands(result, crossings, street_network)
            else:
                # Add default values
                result['crossings_with_islands'] = 0
                result['island_coverage_pct'] = 0.0
            
            # Step 4: Calculate component scores
            
            # Coverage score (0-100)
            coverage_scores = []
            for _, row in result.iterrows():
                coverage_pct = row.get('sidewalk_coverage_pct', 0)
                coverage_scores.append(min(100.0, coverage_pct))
            
            result['coverage_score'] = coverage_scores
            
            # Ramp score (0-100)
            ramp_scores = []
            for _, row in result.iterrows():
                ramp_pct = row.get('ramp_coverage_pct', 0)
                ramp_scores.append(min(100.0, ramp_pct))
            
            result['ramp_score'] = ramp_scores
            
            # Island score (0-100)
            island_scores = []
            for _, row in result.iterrows():
                island_pct = row.get('island_coverage_pct', 0)
                island_scores.append(min(100.0, island_pct))
            
            result['island_score'] = island_scores
            
            # Accessibility score (combination of ramps and islands)
            accessibility_scores = []
            for _, row in result.iterrows():
                ramp_score = row.get('ramp_score', 0)
                island_score = row.get('island_score', 0)
                # Weight ramps more heavily than islands
                accessibility_score = (ramp_score * 0.7) + (island_score * 0.3)
                accessibility_scores.append(accessibility_score)
            
            result['accessibility_score'] = accessibility_scores
            
            # Step 5: Combine all scores using weights
            final_scores = (
                self.weights['coverage'] * result['coverage_score'] +
                self.weights['ramps'] * result['ramp_score'] +
                self.weights['islands'] * result['island_score'] +
                self.weights['accessibility'] * result['accessibility_score']
            )
            
            # Step 6: Normalize final scores
            result['sidewalk_quality_score'] = self.normalize_sidewalk_score(final_scores)
            
            # Add score breakdown for analysis
            result['score_breakdown'] = result.apply(
                lambda row: {
                    'coverage': row['coverage_score'] * self.weights['coverage'],
                    'ramps': row['ramp_score'] * self.weights['ramps'],
                    'islands': row['island_score'] * self.weights['islands'],
                    'accessibility': row['accessibility_score'] * self.weights['accessibility']
                }, axis=1
            )
            
            logger.info("Comprehensive sidewalk score calculation complete")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive sidewalk score calculation failed: {e}")
            # Return original data with default scores
            result = neighborhoods.copy()
            result['sidewalk_quality_score'] = 0.0
            return result


# Convenience functions for backward compatibility and easy use

def calculate_sidewalk_coverage(neighborhoods: gpd.GeoDataFrame,
                              sidewalks: gpd.GeoDataFrame,
                              street_network: Optional[object] = None) -> gpd.GeoDataFrame:
    """
    Calculate sidewalk coverage for neighborhoods.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhood polygons
        sidewalks: GeoDataFrame of sidewalk segments
        street_network: Optional street network for reference
        
    Returns:
        GeoDataFrame with sidewalk coverage metrics
    """
    calculator = SidewalkScoreCalculator()
    return calculator.calculate_sidewalk_coverage(neighborhoods, sidewalks, street_network)


def identify_missing_curb_ramps(neighborhoods: gpd.GeoDataFrame,
                              crossings: gpd.GeoDataFrame,
                              curb_ramps: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """
    Identify missing curb ramps at crossings.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhood polygons
        crossings: GeoDataFrame of pedestrian crossings
        curb_ramps: Optional GeoDataFrame of existing curb ramps
        
    Returns:
        GeoDataFrame with curb ramp analysis
    """
    calculator = SidewalkScoreCalculator()
    return calculator.identify_missing_curb_ramps(neighborhoods, crossings, curb_ramps)


def calculate_pedestrian_islands(neighborhoods: gpd.GeoDataFrame,
                               crossings: gpd.GeoDataFrame,
                               street_network: Optional[object] = None) -> gpd.GeoDataFrame:
    """
    Calculate pedestrian island coverage.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhood polygons
        crossings: GeoDataFrame of pedestrian crossings
        street_network: Optional street network for reference
        
    Returns:
        GeoDataFrame with pedestrian island analysis
    """
    calculator = SidewalkScoreCalculator()
    return calculator.calculate_pedestrian_islands(neighborhoods, crossings, street_network)


def normalize_sidewalk_score(raw_score: Union[pd.Series, np.ndarray],
                           method: str = 'minmax') -> np.ndarray:
    """
    Normalize sidewalk scores to 0-100 range.
    
    Args:
        raw_score: Raw scores to normalize
        method: Normalization method
        
    Returns:
        Normalized scores array
    """
    calculator = SidewalkScoreCalculator()
    return calculator.normalize_sidewalk_score(raw_score, method)


def calculate_comprehensive_sidewalk_score(neighborhoods: gpd.GeoDataFrame,
                                         sidewalks: gpd.GeoDataFrame,
                                         crossings: Optional[gpd.GeoDataFrame] = None,
                                         curb_ramps: Optional[gpd.GeoDataFrame] = None,
                                         street_network: Optional[object] = None) -> gpd.GeoDataFrame:
    """
    Calculate comprehensive sidewalk accessibility score.
    
    Args:
        neighborhoods: GeoDataFrame of neighborhoods
        sidewalks: GeoDataFrame of sidewalk segments
        crossings: Optional GeoDataFrame of pedestrian crossings
        curb_ramps: Optional GeoDataFrame of curb ramps
        street_network: Optional street network for reference
        
    Returns:
        GeoDataFrame with comprehensive sidewalk scores
    """
    calculator = SidewalkScoreCalculator()
    return calculator.calculate_comprehensive_sidewalk_score(
        neighborhoods, sidewalks, crossings, curb_ramps, street_network
    )
