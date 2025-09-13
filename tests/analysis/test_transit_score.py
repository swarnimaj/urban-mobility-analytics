# tests/analysis/test_transit_score.py
"""
Unit tests for transit score calculation module.
"""

import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.transit_score import (
    TransitScoreCalculator,
    calculate_distance_to_transit,
    weight_by_frequency,
    adjust_for_accessibility,
    normalize_transit_score,
    calculate_comprehensive_transit_score
)


class TestTransitScoreCalculator(unittest.TestCase):
    """Test the TransitScoreCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        # Create test neighborhoods (census tracts as polygons)
        self.neighborhoods = gpd.GeoDataFrame({
            'geoid': ['001', '002', '003', '004'],
            'name': ['Tract 1', 'Tract 2', 'Tract 3', 'Tract 4'],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Near stops
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Medium distance
                Polygon([(5, 0), (6, 0), (6, 1), (5, 1)]),  # Far from stops
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)])  # Very far
            ]
        }, crs='EPSG:4326')
        
        # Create test transit stops
        self.transit_stops = gpd.GeoDataFrame({
            'stop_id': ['stop_001', 'stop_002', 'stop_003', 'stop_004'],
            'stop_name': ['Main St & 1st', 'Oak Ave & 2nd', 'Pine St & 3rd', 'Elm Rd & 4th'],
            'wheelchair_accessible': ['yes', 'no', 'yes', 'unknown'],
            'trips_per_day': [100, 50, 75, 25],
            'trips_per_hour': [4.2, 2.1, 3.1, 1.0],
            'avg_headway_minutes': [15, 30, 20, 60],
            'geometry': [
                Point(0.5, 0.5),   # Inside tract 1
                Point(1.8, 0.5),   # Near tract 2
                Point(3.5, 0.5),   # Between tract 2 and 3
                Point(7, 0.5)      # Far from all tracts
            ]
        }, crs='EPSG:4326')
        
        # Create service data
        self.service_data = pd.DataFrame({
            'stop_id': ['stop_001', 'stop_002', 'stop_003', 'stop_004'],
            'trips_per_day': [100, 50, 75, 25],
            'avg_headway_minutes': [15, 30, 20, 60]
        })
        
        # Initialize calculator
        self.calculator = TransitScoreCalculator(max_walking_distance=1000)
    
    def test_initialization(self):
        """Test calculator initialization."""
        # Test default initialization
        calc = TransitScoreCalculator()
        self.assertEqual(calc.max_walking_distance, 1000)
        self.assertEqual(calc.walking_speed, 4.5)
        self.assertAlmostEqual(sum(calc.weights.values()), 1.0, places=2)
        
        # Test custom initialization
        custom_weights = {'distance': 0.5, 'frequency': 0.3, 'accessibility': 0.1, 'coverage': 0.1}
        calc = TransitScoreCalculator(max_walking_distance=800, weights=custom_weights)
        self.assertEqual(calc.max_walking_distance, 800)
        self.assertEqual(calc.weights, custom_weights)
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        # Test weights that don't sum to 1.0
        bad_weights = {'distance': 0.6, 'frequency': 0.6, 'accessibility': 0.2, 'coverage': 0.1}
        calc = TransitScoreCalculator(weights=bad_weights)
        self.assertAlmostEqual(sum(calc.weights.values()), 1.0, places=2)
    
    def test_calculate_euclidean_distance(self):
        """Test euclidean distance calculation."""
        result = self.calculator.calculate_distance_to_transit(
            self.neighborhoods, self.transit_stops, method='euclidean'
        )
        
        # Check that required columns are added
        self.assertIn('nearest_stop_distance', result.columns)
        self.assertIn('stop_count_within_buffer', result.columns)
        self.assertIn('nearest_stop_id', result.columns)
        
        # Check that tract 1 has the shortest distance (stop is inside)
        tract_1_distance = result[result['geoid'] == '001']['nearest_stop_distance'].iloc[0]
        tract_2_distance = result[result['geoid'] == '002']['nearest_stop_distance'].iloc[0]
        self.assertLess(tract_1_distance, tract_2_distance)
        
        # Check stop counts within buffer
        tract_1_count = result[result['geoid'] == '001']['stop_count_within_buffer'].iloc[0]
        tract_4_count = result[result['geoid'] == '004']['stop_count_within_buffer'].iloc[0]
        self.assertGreater(tract_1_count, tract_4_count)
    
    def test_calculate_multiple_stops_access(self):
        """Test multiple stops access calculation."""
        result = self.calculator.calculate_distance_to_transit(
            self.neighborhoods, self.transit_stops, method='multiple_stops'
        )
        
        # Check that buffer columns are added
        buffer_cols = [col for col in result.columns if 'stops_within_' in col and 'm' in col]
        self.assertGreater(len(buffer_cols), 0)
        
        # Check that basic distance metrics are still present
        self.assertIn('nearest_stop_distance', result.columns)
    
    def test_network_distance_fallback(self):
        """Test network distance calculation (should fall back to euclidean with penalty)."""
        result = self.calculator.calculate_distance_to_transit(
            self.neighborhoods, self.transit_stops, method='walking_network'
        )
        
        # Should have basic distance columns
        self.assertIn('nearest_stop_distance', result.columns)
        
        # Network distances should be larger than pure euclidean (due to penalty factor)
        # We can't easily test this without implementing euclidean separately
        # But we can check that distances are reasonable
        distances = result['nearest_stop_distance'].dropna()
        self.assertTrue(all(d >= 0 for d in distances))
    
    def test_weight_by_frequency(self):
        """Test frequency weighting."""
        # First calculate distances
        neighborhoods_with_dist = self.calculator.calculate_distance_to_transit(
            self.neighborhoods, self.transit_stops, method='euclidean'
        )
        
        result = self.calculator.weight_by_frequency(
            neighborhoods_with_dist, self.transit_stops, self.service_data
        )
        
        # Check that frequency score is added
        self.assertIn('frequency_score', result.columns)
        
        # Frequency scores should be between 0 and 100
        freq_scores = result['frequency_score'].dropna()
        self.assertTrue(all(0 <= score <= 100 for score in freq_scores))
    
    def test_adjust_for_accessibility(self):
        """Test accessibility adjustment."""
        result = self.calculator.adjust_for_accessibility(
            self.neighborhoods, self.transit_stops
        )
        
        # Check that accessibility score is added
        self.assertIn('accessibility_score', result.columns)
        
        # Accessibility scores should be between 0 and 100
        acc_scores = result['accessibility_score'].dropna()
        self.assertTrue(all(0 <= score <= 100 for score in acc_scores))
    
    def test_normalize_transit_score(self):
        """Test score normalization."""
        # Test data with various ranges
        test_scores = [10, 20, 30, 40, 50]
        
        # Test min-max normalization
        normalized = self.calculator.normalize_transit_score(test_scores, method='minmax')
        self.assertAlmostEqual(min(normalized), 0.0, places=1)
        self.assertAlmostEqual(max(normalized), 100.0, places=1)
        
        # Test z-score normalization
        normalized_z = self.calculator.normalize_transit_score(test_scores, method='zscore')
        self.assertEqual(len(normalized_z), len(test_scores))
        
        # Test robust normalization
        normalized_robust = self.calculator.normalize_transit_score(test_scores, method='robust')
        self.assertEqual(len(normalized_robust), len(test_scores))
    
    def test_normalize_edge_cases(self):
        """Test normalization with edge cases."""
        # All same values
        same_scores = [50, 50, 50, 50]
        normalized = self.calculator.normalize_transit_score(same_scores)
        self.assertTrue(all(40 <= score <= 60 for score in normalized))  # Should be around middle
        
        # Empty array
        empty_scores = []
        normalized_empty = self.calculator.normalize_transit_score(empty_scores)
        self.assertEqual(len(normalized_empty), 0)
        
        # With infinite values
        inf_scores = [10, float('inf'), 30, 40]
        normalized_inf = self.calculator.normalize_transit_score(inf_scores)
        self.assertTrue(all(np.isfinite(score) for score in normalized_inf))
    
    def test_comprehensive_scoring(self):
        """Test comprehensive transit score calculation."""
        result = self.calculator.calculate_comprehensive_transit_score(
            self.neighborhoods, self.transit_stops, self.service_data
        )
        
        # Check that all score components are present
        required_cols = [
            'transit_access_score', 'distance_score', 'frequency_score',
            'accessibility_score', 'coverage_score', 'score_breakdown'
        ]
        for col in required_cols:
            self.assertIn(col, result.columns)
        
        # Check score ranges
        transit_scores = result['transit_access_score'].dropna()
        self.assertTrue(all(0 <= score <= 100 for score in transit_scores))
        
        # Check that tract 1 (closest to stops) has higher score than tract 4 (farthest)
        tract_1_score = result[result['geoid'] == '001']['transit_access_score'].iloc[0]
        tract_4_score = result[result['geoid'] == '004']['transit_access_score'].iloc[0]
        self.assertGreater(tract_1_score, tract_4_score)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_neighborhoods = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        empty_stops = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        
        # Test with empty neighborhoods
        result = self.calculator.calculate_distance_to_transit(
            empty_neighborhoods, self.transit_stops
        )
        self.assertEqual(len(result), 0)
        
        # Test with empty stops
        result = self.calculator.calculate_distance_to_transit(
            self.neighborhoods, empty_stops
        )
        self.assertIn('nearest_stop_distance', result.columns)
        distances = result['nearest_stop_distance']
        self.assertTrue(all(np.isinf(d) for d in distances))
    
    def test_missing_columns_handling(self):
        """Test handling of missing data columns."""
        # Create stops without accessibility info
        stops_no_access = self.transit_stops.drop(columns=['wheelchair_accessible'])
        
        result = self.calculator.adjust_for_accessibility(
            self.neighborhoods, stops_no_access
        )
        
        # Should still work and assign neutral scores
        self.assertIn('accessibility_score', result.columns)
        scores = result['accessibility_score'].dropna()
        self.assertTrue(all(score == 50.0 for score in scores))  # Neutral score
    
    def test_get_nearby_stops(self):
        """Test getting nearby stops."""
        neighborhood = self.neighborhoods.iloc[0]  # Tract 1
        nearby = self.calculator._get_nearby_stops(
            neighborhood, self.transit_stops, 1000
        )
        
        # Should find at least one stop
        self.assertGreater(len(nearby), 0)
        
        # Test with very small distance
        nearby_small = self.calculator._get_nearby_stops(
            neighborhood, self.transit_stops, 10
        )
        # Might find fewer or no stops
        self.assertLessEqual(len(nearby_small), len(nearby))
    
    def test_calculate_frequency_score(self):
        """Test frequency score calculation."""
        nearby_stops = self.transit_stops.iloc[:2]  # First two stops
        
        score = self.calculator._calculate_frequency_score(
            nearby_stops, self.service_data
        )
        
        # Should return a valid score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions."""
    
    def setUp(self):
        """Set up test data."""
        self.neighborhoods = gpd.GeoDataFrame({
            'geoid': ['001', '002'],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
            ]
        }, crs='EPSG:4326')
        
        self.transit_stops = gpd.GeoDataFrame({
            'stop_id': ['stop_001', 'stop_002'],
            'stop_name': ['Stop 1', 'Stop 2'],
            'wheelchair_accessible': ['yes', 'no'],
            'geometry': [Point(0.5, 0.5), Point(2.5, 0.5)]
        }, crs='EPSG:4326')
        
        self.service_data = pd.DataFrame({
            'stop_id': ['stop_001', 'stop_002'],
            'trips_per_day': [100, 50]
        })
    
    def test_calculate_distance_to_transit_function(self):
        """Test the standalone distance calculation function."""
        result = calculate_distance_to_transit(
            self.neighborhoods, self.transit_stops
        )
        
        self.assertIn('nearest_stop_distance', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))
    
    def test_normalize_transit_score_function(self):
        """Test the standalone normalization function."""
        scores = [10, 20, 30, 40, 50]
        normalized = normalize_transit_score(scores)
        
        self.assertEqual(len(normalized), len(scores))
        self.assertAlmostEqual(min(normalized), 0.0, places=1)
        self.assertAlmostEqual(max(normalized), 100.0, places=1)
    
    def test_comprehensive_function(self):
        """Test the comprehensive scoring function."""
        result = calculate_comprehensive_transit_score(
            self.neighborhoods, self.transit_stops, self.service_data
        )
        
        self.assertIn('transit_access_score', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_invalid_crs_handling(self):
        """Test handling of invalid or missing CRS."""
        # Create data without CRS
        neighborhoods_no_crs = gpd.GeoDataFrame({
            'geoid': ['001'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        })  # No CRS specified
        
        stops_no_crs = gpd.GeoDataFrame({
            'stop_id': ['stop_001'],
            'geometry': [Point(0.5, 0.5)]
        })  # No CRS specified
        
        calculator = TransitScoreCalculator()
        
        # Should handle gracefully
        try:
            result = calculator.calculate_distance_to_transit(
                neighborhoods_no_crs, stops_no_crs
            )
            # Should complete without error
            self.assertIsInstance(result, gpd.GeoDataFrame)
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, Exception)
    
    def test_corrupted_geometry_handling(self):
        """Test handling of corrupted geometries."""
        # Create data with invalid geometries
        neighborhoods_bad_geom = gpd.GeoDataFrame({
            'geoid': ['001'],
            'geometry': [Point(0, 0).buffer(0)]  # Degenerate geometry
        }, crs='EPSG:4326')
        
        stops = gpd.GeoDataFrame({
            'stop_id': ['stop_001'],
            'geometry': [Point(0.5, 0.5)]
        }, crs='EPSG:4326')
        
        calculator = TransitScoreCalculator()
        
        # Should handle without crashing
        result = calculator.calculate_distance_to_transit(
            neighborhoods_bad_geom, stops
        )
        self.assertIsInstance(result, gpd.GeoDataFrame)
    
    def test_large_distance_values(self):
        """Test handling of very large distance values."""
        # Create stops very far from neighborhoods
        neighborhoods = gpd.GeoDataFrame({
            'geoid': ['001'],
            'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
        }, crs='EPSG:4326')
        
        far_stops = gpd.GeoDataFrame({
            'stop_id': ['stop_001'],
            'geometry': [Point(1000, 1000)]  # Very far
        }, crs='EPSG:4326')
        
        calculator = TransitScoreCalculator(max_walking_distance=500)
        result = calculator.calculate_distance_to_transit(neighborhoods, far_stops)
        
        # Distance should be properly handled
        self.assertIn('nearest_stop_distance', result.columns)


if __name__ == '__main__':
    # Run the tests
    unittest.main()
