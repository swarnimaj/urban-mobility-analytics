# tests/analysis/test_sidewalk_score.py
"""
Unit tests for sidewalk scoring functions.
"""

import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import sys
from pathlib import Path

# Add the src directory to the path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.analysis.sidewalk_score import (
    SidewalkScoreCalculator,
    calculate_sidewalk_coverage,
    identify_missing_curb_ramps,
    calculate_pedestrian_islands,
    normalize_sidewalk_score,
    calculate_comprehensive_sidewalk_score
)


class TestSidewalkScoreCalculator(unittest.TestCase):
    """Test cases for SidewalkScoreCalculator class."""
    
    def setUp(self):
        """Set up test data."""
        # Create test neighborhoods
        self.neighborhoods = gpd.GeoDataFrame({
            'geoid': ['001', '002', '003'],
            'name': ['Neighborhood A', 'Neighborhood B', 'Neighborhood C']
        }, geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])
        ], crs="EPSG:4326")
        
        # Create test sidewalks
        self.sidewalks = gpd.GeoDataFrame({
            'sidewalk_id': ['sw1', 'sw2', 'sw3'],
            'infrastructure_type': ['sidewalk', 'sidewalk', 'sidewalk']
        }, geometry=[
            LineString([(0.1, 0.1), (0.9, 0.1)]),
            LineString([(2.1, 0.1), (2.9, 0.1)]),
            LineString([(0.1, 2.1), (0.9, 2.1)])
        ], crs="EPSG:4326")
        
        # Create test crossings
        self.crossings = gpd.GeoDataFrame({
            'crossing_id': ['c1', 'c2', 'c3'],
            'crossing': ['marked', 'unmarked', 'marked']
        }, geometry=[
            Point(0.5, 0.5),
            Point(2.5, 0.5),
            Point(0.5, 2.5)
        ], crs="EPSG:4326")
        
        # Create test curb ramps
        self.curb_ramps = gpd.GeoDataFrame({
            'ramp_id': ['r1', 'r2'],
            'kerb': ['lowered', 'flush']
        }, geometry=[
            Point(0.5, 0.4),
            Point(2.5, 0.4)
        ], crs="EPSG:4326")
        
        self.calculator = SidewalkScoreCalculator()
    
    def test_initialization(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.sidewalk_width, 1.5)
        self.assertEqual(self.calculator.crossing_width, 3.0)
        self.assertIn('coverage', self.calculator.weights)
        self.assertIn('ramps', self.calculator.weights)
        self.assertIn('islands', self.calculator.weights)
        self.assertIn('accessibility', self.calculator.weights)
    
    def test_calculate_sidewalk_coverage(self):
        """Test sidewalk coverage calculation."""
        result = self.calculator.calculate_sidewalk_coverage(
            self.neighborhoods, self.sidewalks
        )
        
        # Check that result has expected columns
        expected_cols = ['sidewalk_coverage_pct', 'sidewalk_length_km', 'sidewalk_density_km_sqkm']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that all neighborhoods have coverage data
        self.assertEqual(len(result), len(self.neighborhoods))
        
        # Check that coverage percentages are reasonable
        coverage_pcts = result['sidewalk_coverage_pct']
        self.assertTrue(all(0 <= pct <= 100 for pct in coverage_pcts))
    
    def test_calculate_sidewalk_coverage_empty_sidewalks(self):
        """Test sidewalk coverage with empty sidewalk data."""
        empty_sidewalks = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = self.calculator.calculate_sidewalk_coverage(
            self.neighborhoods, empty_sidewalks
        )
        
        # All coverage should be 0
        self.assertTrue(all(result['sidewalk_coverage_pct'] == 0))
        self.assertTrue(all(result['sidewalk_length_km'] == 0))
        self.assertTrue(all(result['sidewalk_density_km_sqkm'] == 0))
    
    def test_identify_missing_curb_ramps(self):
        """Test curb ramp identification."""
        result = self.calculator.identify_missing_curb_ramps(
            self.neighborhoods, self.crossings, self.curb_ramps
        )
        
        # Check that result has expected columns
        expected_cols = ['total_crossings', 'crossings_with_ramps', 'ramp_coverage_pct']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that all neighborhoods have ramp data
        self.assertEqual(len(result), len(self.neighborhoods))
        
        # Check that coverage percentages are reasonable
        ramp_pcts = result['ramp_coverage_pct']
        self.assertTrue(all(0 <= pct <= 100 for pct in ramp_pcts))
    
    def test_identify_missing_curb_ramps_no_crossings(self):
        """Test curb ramp identification with no crossings."""
        empty_crossings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = self.calculator.identify_missing_curb_ramps(
            self.neighborhoods, empty_crossings
        )
        
        # All values should be 0
        self.assertTrue(all(result['total_crossings'] == 0))
        self.assertTrue(all(result['crossings_with_ramps'] == 0))
        self.assertTrue(all(result['ramp_coverage_pct'] == 0))
    
    def test_calculate_pedestrian_islands(self):
        """Test pedestrian island calculation."""
        result = self.calculator.calculate_pedestrian_islands(
            self.neighborhoods, self.crossings
        )
        
        # Check that result has expected columns
        expected_cols = ['total_crossings', 'crossings_with_islands', 'island_coverage_pct']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that all neighborhoods have island data
        self.assertEqual(len(result), len(self.neighborhoods))
        
        # Check that coverage percentages are reasonable
        island_pcts = result['island_coverage_pct']
        self.assertTrue(all(0 <= pct <= 100 for pct in island_pcts))
    
    def test_normalize_sidewalk_score(self):
        """Test score normalization."""
        # Test with simple array
        raw_scores = np.array([10, 20, 30, 40, 50])
        normalized = self.calculator.normalize_sidewalk_score(raw_scores)
        
        # Check that normalized scores are in expected range
        self.assertTrue(all(0 <= score <= 100 for score in normalized))
        
        # Check that min and max are preserved (approximately)
        self.assertAlmostEqual(normalized.min(), 0, places=1)
        self.assertAlmostEqual(normalized.max(), 100, places=1)
    
    def test_normalize_sidewalk_score_edge_cases(self):
        """Test score normalization with edge cases."""
        # Test with all same values
        same_scores = np.array([50, 50, 50, 50])
        normalized = self.calculator.normalize_sidewalk_score(same_scores)
        self.assertTrue(all(score == 50 for score in normalized))
        
        # Test with empty array
        empty_scores = np.array([])
        normalized = self.calculator.normalize_sidewalk_score(empty_scores)
        self.assertEqual(len(normalized), 0)
        
        # Test with NaN values
        nan_scores = np.array([10, np.nan, 30, np.nan, 50])
        normalized = self.calculator.normalize_sidewalk_score(nan_scores)
        self.assertEqual(len(normalized), len(nan_scores))
    
    def test_validate_sidewalk_data(self):
        """Test sidewalk data validation."""
        validation = self.calculator.validate_sidewalk_data(
            self.sidewalks, self.crossings, self.curb_ramps
        )
        
        # Check that validation has expected structure
        self.assertIn('sidewalks', validation)
        self.assertIn('crossings', validation)
        self.assertIn('curb_ramps', validation)
        self.assertIn('overall_quality', validation)
        self.assertIn('overall_score', validation)
        
        # Check that quality scores are reasonable
        self.assertTrue(0 <= validation['overall_score'] <= 100)
        self.assertIn(validation['overall_quality'], ['excellent', 'good', 'fair', 'poor'])
    
    def test_validate_sidewalk_data_empty(self):
        """Test validation with empty data."""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        validation = self.calculator.validate_sidewalk_data(empty_gdf, empty_gdf, empty_gdf)
        
        # All quality scores should be 0
        self.assertEqual(validation['sidewalks']['quality_score'], 0)
        self.assertEqual(validation['crossings']['quality_score'], 0)
        self.assertEqual(validation['curb_ramps']['quality_score'], 0)
        self.assertEqual(validation['overall_score'], 0)
    
    def test_calculate_comprehensive_sidewalk_score(self):
        """Test comprehensive sidewalk score calculation."""
        result = self.calculator.calculate_comprehensive_sidewalk_score(
            self.neighborhoods, self.sidewalks, self.crossings, self.curb_ramps
        )
        
        # Check that result has expected columns
        expected_cols = [
            'sidewalk_quality_score', 'coverage_score', 'ramp_score', 
            'island_score', 'accessibility_score', 'score_breakdown'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        # Check that all neighborhoods have scores
        self.assertEqual(len(result), len(self.neighborhoods))
        
        # Check that scores are in expected range
        quality_scores = result['sidewalk_quality_score']
        self.assertTrue(all(0 <= score <= 100 for score in quality_scores))
    
    def test_calculate_comprehensive_sidewalk_score_minimal_data(self):
        """Test comprehensive scoring with minimal data."""
        result = self.calculator.calculate_comprehensive_sidewalk_score(
            self.neighborhoods, self.sidewalks
        )
        
        # Should still work with minimal data
        self.assertIn('sidewalk_quality_score', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test data."""
        self.neighborhoods = gpd.GeoDataFrame({
            'geoid': ['001', '002']
        }, geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        ], crs="EPSG:4326")
        
        self.sidewalks = gpd.GeoDataFrame({
            'sidewalk_id': ['sw1', 'sw2']
        }, geometry=[
            LineString([(0.1, 0.1), (0.9, 0.1)]),
            LineString([(2.1, 0.1), (2.9, 0.1)])
        ], crs="EPSG:4326")
        
        self.crossings = gpd.GeoDataFrame({
            'crossing_id': ['c1', 'c2']
        }, geometry=[
            Point(0.5, 0.5),
            Point(2.5, 0.5)
        ], crs="EPSG:4326")
    
    def test_calculate_sidewalk_coverage_function(self):
        """Test calculate_sidewalk_coverage convenience function."""
        result = calculate_sidewalk_coverage(self.neighborhoods, self.sidewalks)
        
        self.assertIn('sidewalk_coverage_pct', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))
    
    def test_identify_missing_curb_ramps_function(self):
        """Test identify_missing_curb_ramps convenience function."""
        result = identify_missing_curb_ramps(self.neighborhoods, self.crossings)
        
        self.assertIn('ramp_coverage_pct', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))
    
    def test_calculate_pedestrian_islands_function(self):
        """Test calculate_pedestrian_islands convenience function."""
        result = calculate_pedestrian_islands(self.neighborhoods, self.crossings)
        
        self.assertIn('island_coverage_pct', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))
    
    def test_normalize_sidewalk_score_function(self):
        """Test normalize_sidewalk_score convenience function."""
        raw_scores = np.array([10, 20, 30, 40, 50])
        normalized = normalize_sidewalk_score(raw_scores)
        
        self.assertEqual(len(normalized), len(raw_scores))
        self.assertTrue(all(0 <= score <= 100 for score in normalized))
    
    def test_calculate_comprehensive_sidewalk_score_function(self):
        """Test calculate_comprehensive_sidewalk_score convenience function."""
        result = calculate_comprehensive_sidewalk_score(
            self.neighborhoods, self.sidewalks, self.crossings
        )
        
        self.assertIn('sidewalk_quality_score', result.columns)
        self.assertEqual(len(result), len(self.neighborhoods))


if __name__ == '__main__':
    unittest.main()
