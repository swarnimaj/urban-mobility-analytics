"""
Unit tests for amenity scoring functionality - Sprint 9
"""

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.amenity_score import (
    AmenityScoreCalculator,
    calculate_amenity_distances,
    weight_by_amenity_type,
    calculate_accessibility_penalty,
    normalize_amenity_score
)


class TestAmenityScoreCalculator(unittest.TestCase):
    """Test cases for AmenityScoreCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        # Create test neighborhoods
        self.neighborhoods = gpd.GeoDataFrame({
            'neighborhood_id': ['N1', 'N2', 'N3'],
            'name': ['Downtown', 'Suburb', 'Rural']
        }, geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)])
        ], crs='EPSG:4326')
        
        # Create test amenities
        self.amenities = gpd.GeoDataFrame({
            'amenity_id': ['A1', 'A2', 'A3', 'A4'],
            'amenity': ['hospital', 'school', 'restaurant', 'cafe'],
            'name': ['City Hospital', 'Central School', 'Main Restaurant', 'Coffee Shop'],
            'accessibility_score': [4.0, 2.0, 1.0, 1.0],
            'wheelchair_score': [3.0, 2.0, 1.0, 1.0],
            'accessibility_category': ['fully_accessible', 'partially_accessible', 'not_accessible', 'unknown']
        }, geometry=[
            Point(0.5, 0.5),  # Inside N1
            Point(2.5, 0.5),  # Inside N2
            Point(0.2, 0.2),  # Inside N1
            Point(4.5, 0.5)   # Inside N3
        ], crs='EPSG:4326')
        
        # Initialize calculator
        self.calculator = AmenityScoreCalculator(max_distance=1000.0)
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = AmenityScoreCalculator(max_distance=1500.0, distance_method='euclidean')
        self.assertEqual(calculator.max_distance, 1500.0)
        self.assertEqual(calculator.distance_method, 'euclidean')
        self.assertEqual(calculator.accessibility_penalty, 0.5)
        self.assertEqual(calculator.normalization_method, 'minmax')
    
    def test_calculate_amenity_distances(self):
        """Test distance calculation."""
        distances_df = self.calculator.calculate_amenity_distances(self.neighborhoods, self.amenities)
        
        # Check that we get distance data
        self.assertFalse(distances_df.empty)
        self.assertIn('neighborhood_id', distances_df.columns)
        self.assertIn('amenity_id', distances_df.columns)
        self.assertIn('distance_m', distances_df.columns)
        self.assertIn('amenity_type', distances_df.columns)
        
        # Check that distances are reasonable
        self.assertTrue((distances_df['distance_m'] >= 0).all())
        self.assertTrue((distances_df['distance_m'] <= self.calculator.max_distance).all())
    
    def test_calculate_amenity_distances_empty_amenities(self):
        """Test distance calculation with empty amenities."""
        empty_amenities = gpd.GeoDataFrame(columns=['amenity', 'geometry'], crs='EPSG:4326')
        distances_df = self.calculator.calculate_amenity_distances(self.neighborhoods, empty_amenities)
        self.assertTrue(distances_df.empty)
    
    def test_weight_by_amenity_type(self):
        """Test amenity type weighting."""
        # Create sample distance data
        distances_df = pd.DataFrame({
            'neighborhood_id': ['N1', 'N1', 'N2'],
            'amenity_id': ['A1', 'A2', 'A1'],
            'amenity_type': ['hospital', 'cafe', 'hospital'],
            'distance_m': [100.0, 200.0, 150.0],
            'accessibility_score': [4.0, 1.0, 4.0],
            'wheelchair_score': [3.0, 1.0, 3.0],
            'accessibility_category': ['fully_accessible', 'unknown', 'fully_accessible']
        })
        
        weighted_df = self.calculator.weight_by_amenity_type(distances_df)
        
        # Check that weights are applied
        self.assertIn('amenity_weight', weighted_df.columns)
        self.assertIn('distance_score', weighted_df.columns)
        self.assertIn('accessibility_weight', weighted_df.columns)
        self.assertIn('weighted_score', weighted_df.columns)
        
        # Check that hospital has higher weight than cafe
        hospital_row = weighted_df[weighted_df['amenity_type'] == 'hospital'].iloc[0]
        cafe_row = weighted_df[weighted_df['amenity_type'] == 'cafe'].iloc[0]
        self.assertGreater(hospital_row['amenity_weight'], cafe_row['amenity_weight'])
    
    def test_calculate_accessibility_penalty(self):
        """Test accessibility penalty calculation."""
        # Create sample weighted data
        weighted_df = pd.DataFrame({
            'neighborhood_id': ['N1', 'N1', 'N2'],
            'amenity_id': ['A1', 'A2', 'A1'],
            'amenity_type': ['hospital', 'cafe', 'hospital'],
            'distance_m': [100.0, 200.0, 150.0],
            'weighted_score': [0.8, 0.4, 0.7],
            'accessibility_category': ['fully_accessible', 'not_accessible', 'partially_accessible']
        })
        
        penalty_df = self.calculator.calculate_accessibility_penalty(weighted_df)
        
        # Check that penalties are applied
        self.assertIn('accessibility_penalty', penalty_df.columns)
        self.assertIn('final_score', penalty_df.columns)
        self.assertIn('penalized_distance', penalty_df.columns)
        
        # Check that not_accessible has higher penalty than fully_accessible
        not_accessible_row = penalty_df[penalty_df['accessibility_category'] == 'not_accessible'].iloc[0]
        fully_accessible_row = penalty_df[penalty_df['accessibility_category'] == 'fully_accessible'].iloc[0]
        self.assertGreater(not_accessible_row['accessibility_penalty'], fully_accessible_row['accessibility_penalty'])
    
    def test_normalize_amenity_score(self):
        """Test score normalization."""
        raw_scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = self.calculator.normalize_amenity_score(raw_scores)
        
        # Check that scores are in 0-100 range
        self.assertTrue((normalized >= 0).all())
        self.assertTrue((normalized <= 100).all())
        
        # Check that normalization preserves order
        self.assertTrue(np.all(np.diff(normalized) >= 0))
    
    def test_normalize_amenity_score_single_value(self):
        """Test normalization with single value."""
        raw_scores = np.array([25.0])
        normalized = self.calculator.normalize_amenity_score(raw_scores)
        
        # Single value should be normalized to 50 (middle of range)
        self.assertEqual(normalized[0], 50.0)
    
    def test_normalize_amenity_score_empty(self):
        """Test normalization with empty array."""
        raw_scores = np.array([])
        normalized = self.calculator.normalize_amenity_score(raw_scores)
        
        self.assertEqual(len(normalized), 0)
    
    def test_calculate_comprehensive_amenity_score(self):
        """Test comprehensive score calculation."""
        result = self.calculator.calculate_comprehensive_amenity_score(self.neighborhoods, self.amenities)
        
        # Check that result has expected columns
        self.assertIn('amenity_access_score', result.columns)
        self.assertIn('amenity_density', result.columns)
        self.assertIn('amenity_diversity', result.columns)
        self.assertIn('accessibility_score', result.columns)
        
        # Check that scores are in valid range
        self.assertTrue((result['amenity_access_score'] >= 0).all())
        self.assertTrue((result['amenity_access_score'] <= 100).all())
    
    def test_calculate_comprehensive_amenity_score_no_amenities(self):
        """Test comprehensive scoring with no amenities in range."""
        # Create amenities far from neighborhoods
        far_amenities = gpd.GeoDataFrame({
            'amenity': ['hospital'],
            'name': ['Far Hospital'],
            'accessibility_score': [4.0],
            'wheelchair_score': [3.0],
            'accessibility_category': ['fully_accessible']
        }, geometry=[Point(100, 100)], crs='EPSG:4326')
        
        result = self.calculator.calculate_comprehensive_amenity_score(self.neighborhoods, far_amenities)
        
        # All scores should be 0
        self.assertTrue((result['amenity_access_score'] == 0).all())
    
    def test_validate_amenity_data(self):
        """Test amenity data validation."""
        validation = self.calculator.validate_amenity_data(self.amenities)
        
        # Check validation results
        self.assertIn('total_amenities', validation)
        self.assertIn('amenity_types', validation)
        self.assertIn('has_geometry', validation)
        self.assertIn('has_accessibility', validation)
        self.assertIn('overall_quality_score', validation)
        
        # Check that validation results are reasonable
        self.assertEqual(validation['total_amenities'], len(self.amenities))
        self.assertTrue(validation['has_geometry'])
        self.assertTrue(validation['has_accessibility'])
        self.assertGreater(validation['overall_quality_score'], 0)
    
    def test_validate_amenity_data_empty(self):
        """Test validation with empty amenity data."""
        empty_amenities = gpd.GeoDataFrame(columns=['amenity', 'geometry'], crs='EPSG:4326')
        validation = self.calculator.validate_amenity_data(empty_amenities)
        
        self.assertEqual(validation['total_amenities'], 0)
        self.assertEqual(validation['overall_quality_score'], 0.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test data."""
        self.neighborhoods = gpd.GeoDataFrame({
            'neighborhood_id': ['N1', 'N2']
        }, geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        ], crs='EPSG:4326')
        
        self.amenities = gpd.GeoDataFrame({
            'amenity': ['hospital', 'school'],
            'name': ['Hospital', 'School'],
            'accessibility_score': [4.0, 2.0],
            'wheelchair_score': [3.0, 2.0],
            'accessibility_category': ['fully_accessible', 'partially_accessible']
        }, geometry=[
            Point(0.5, 0.5),
            Point(2.5, 0.5)
        ], crs='EPSG:4326')
    
    def test_calculate_amenity_distances_function(self):
        """Test calculate_amenity_distances convenience function."""
        distances_df = calculate_amenity_distances(self.neighborhoods, self.amenities)
        
        self.assertFalse(distances_df.empty)
        self.assertIn('distance_m', distances_df.columns)
        self.assertIn('amenity_type', distances_df.columns)
    
    def test_weight_by_amenity_type_function(self):
        """Test weight_by_amenity_type convenience function."""
        distances_df = pd.DataFrame({
            'amenity_type': ['hospital', 'cafe'],
            'distance_m': [100.0, 200.0],
            'accessibility_category': ['fully_accessible', 'unknown']
        })
        
        weighted_df = weight_by_amenity_type(distances_df)
        
        self.assertIn('weighted_score', weighted_df.columns)
        self.assertIn('amenity_weight', weighted_df.columns)
    
    def test_calculate_accessibility_penalty_function(self):
        """Test calculate_accessibility_penalty convenience function."""
        weighted_df = pd.DataFrame({
            'weighted_score': [0.8, 0.4],
            'accessibility_category': ['fully_accessible', 'not_accessible']
        })
        
        penalty_df = calculate_accessibility_penalty(weighted_df)
        
        self.assertIn('final_score', penalty_df.columns)
        self.assertIn('accessibility_penalty', penalty_df.columns)
    
    def test_normalize_amenity_score_function(self):
        """Test normalize_amenity_score convenience function."""
        raw_scores = np.array([10.0, 20.0, 30.0])
        normalized = normalize_amenity_score(raw_scores)
        
        self.assertTrue((normalized >= 0).all())
        self.assertTrue((normalized <= 100).all())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_calculator_with_invalid_distance_method(self):
        """Test calculator with invalid distance method."""
        with self.assertRaises(ValueError):
            calculator = AmenityScoreCalculator(distance_method='invalid')
    
    def test_normalize_with_all_nan_values(self):
        """Test normalization with all NaN values."""
        calculator = AmenityScoreCalculator()
        raw_scores = np.array([np.nan, np.nan, np.nan])
        normalized = calculator.normalize_amenity_score(raw_scores)
        
        # Should return zeros for NaN values
        self.assertTrue(np.allclose(normalized, 0.0))
    
    def test_normalize_with_invalid_method(self):
        """Test normalization with invalid method."""
        calculator = AmenityScoreCalculator()
        raw_scores = np.array([10.0, 20.0, 30.0])
        
        with self.assertRaises(ValueError):
            calculator.normalize_amenity_score(raw_scores, method='invalid')
    
    def test_calculator_with_crs_mismatch(self):
        """Test calculator with CRS mismatch."""
        neighborhoods = gpd.GeoDataFrame({
            'neighborhood_id': ['N1']
        }, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:4326')
        
        amenities = gpd.GeoDataFrame({
            'amenity': ['hospital'],
            'name': ['Hospital'],
            'accessibility_score': [4.0],
            'wheelchair_score': [3.0],
            'accessibility_category': ['fully_accessible']
        }, geometry=[Point(0.5, 0.5)], crs='EPSG:3857')  # Different CRS
        
        calculator = AmenityScoreCalculator()
        
        # Should handle CRS mismatch gracefully
        distances_df = calculator.calculate_amenity_distances(neighborhoods, amenities)
        self.assertIsInstance(distances_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
