"""
Unit tests for the osm_downloader module.
"""
import unittest
import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
from pathlib import Path
import sys
from shapely.geometry import Polygon, Point
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_acquisition.osm_downloader import OSMDownloader

class TestOSMDownloader(unittest.TestCase):
    """Test the OSM downloader functionality."""
    
    def setUp(self):
        """Set up test data and downloader."""
        # Use Bellevue for testing - good OSM coverage, manageable size
        self.test_place = "Bellevue, Washington"
        
        # Test area: ~4-5 km² in Bellevue (47.60-47.63°N, 122.15-122.20°W)
        # Large enough for meaningful data, small enough for fast tests
        self.test_polygon = Polygon([
            (-122.200, 47.605),
            (-122.200, 47.620),
            (-122.185, 47.620),
            (-122.185, 47.605),
            (-122.200, 47.605)
        ])
        
        # Use temp directory for test cache
        self.temp_cache_dir = tempfile.mkdtemp()
        
        # Initialize downloader
        self.downloader = OSMDownloader(
            place_name=self.test_place,
            cache_folder=self.temp_cache_dir
        )
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_cache_dir):
            shutil.rmtree(self.temp_cache_dir)
    
    def test_initialization(self):
        """Test basic initialization with place name."""
        # Verify boundary was created
        self.assertIsNotNone(self.downloader.boundary)
        self.assertIsInstance(self.downloader.boundary, gpd.GeoDataFrame)
        
        # Verify settings
        self.assertEqual(self.downloader.place_name, self.test_place)
        self.assertEqual(self.downloader.cache_folder, Path(self.temp_cache_dir))
    
    def test_initialization_with_polygon(self):
        """Test initialization with custom boundary polygon."""
        downloader = OSMDownloader(boundary=self.test_polygon)
        
        self.assertEqual(downloader.boundary, self.test_polygon)
        self.assertIsNone(downloader.place_name)
    
    def test_initialization_with_custom_cache(self):
        """Test initialization with custom cache directory."""
        custom_cache = tempfile.mkdtemp()
        try:
            downloader = OSMDownloader(
                place_name=self.test_place,
                cache_folder=custom_cache
            )
            self.assertEqual(downloader.cache_folder, Path(custom_cache))
        finally:
            shutil.rmtree(custom_cache)
    
    def test_get_street_network(self):
        """Test street network download."""
        # Download a small network
        G = self.downloader.get_street_network(network_type='drive')
        
        # Verify we got a valid network
        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertGreater(len(G.nodes), 0)
        self.assertGreater(len(G.edges), 0)
    
    def test_get_street_network_with_polygon(self):
        """Test getting a street network with a polygon."""
        # Get a small street network using the test polygon
        G = self.downloader.get_street_network(boundary=self.test_polygon, network_type='drive')
        
        # Check that we got a NetworkX graph
        self.assertIsInstance(G, nx.MultiDiGraph)
        
        # Check that it has nodes and edges
        self.assertGreater(len(G.nodes), 0)
        self.assertGreater(len(G.edges), 0)
    
    def test_get_street_network_caching(self):
        """Test that street network caching works."""
        # First call should download
        G1 = self.downloader.get_street_network(network_type='drive', cache=True)
        
        # Second call should load from cache
        G2 = self.downloader.get_street_network(network_type='drive', cache=True)
        
        # Both should be the same
        self.assertEqual(len(G1.nodes), len(G2.nodes))
        self.assertEqual(len(G1.edges), len(G2.edges))
    
    def test_get_sidewalk_data(self):
        """Test getting sidewalk data."""
        # Get sidewalk data
        sidewalks = self.downloader.get_sidewalk_data()
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(sidewalks, gpd.GeoDataFrame)
        
        # Check that it has the infrastructure_type column
        self.assertIn('infrastructure_type', sidewalks.columns)
        
        # Check that it has some sidewalks or crossings
        if not sidewalks.empty:
            types = sidewalks['infrastructure_type'].unique()
            self.assertTrue(any(t in types for t in ['sidewalk', 'crossing', 'curb_ramp', 'other']))
    
    def test_get_sidewalk_data_with_polygon(self):
        """Test getting sidewalk data with a polygon."""
        # Get sidewalk data using the test polygon
        sidewalks = self.downloader.get_sidewalk_data(boundary=self.test_polygon)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(sidewalks, gpd.GeoDataFrame)
        
        # Check that it has the infrastructure_type column
        self.assertIn('infrastructure_type', sidewalks.columns)
    
    def test_get_amenities(self):
        """Test getting amenities."""
        # Get amenities (just schools for faster test)
        amenities = self.downloader.get_amenities(amenity_types=['school'])
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(amenities, gpd.GeoDataFrame)
        
        # Check that it has the amenity column
        self.assertIn('amenity', amenities.columns)
        
        # Check that all amenities are schools (if any found)
        if not amenities.empty:
            self.assertTrue(all(amenities['amenity'] == 'school'))
        else:
            # If no schools found in the small test area, that's okay
            # Try with a broader area or different amenity type
            amenities = self.downloader.get_amenities(amenity_types=['restaurant'])
            self.assertIsInstance(amenities, gpd.GeoDataFrame)
            self.assertIn('amenity', amenities.columns)
    
    def test_get_amenities_default_types(self):
        """Test getting amenities with default types."""
        # Get amenities with default types
        amenities = self.downloader.get_amenities()
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(amenities, gpd.GeoDataFrame)
        
        # Check that it has the amenity column
        self.assertIn('amenity', amenities.columns)
    
    def test_filter_accessible_amenities(self):
        """Test filtering accessible amenities."""
        # Get amenities with multiple types to increase chances of finding some
        amenities = self.downloader.get_amenities(amenity_types=['school', 'hospital', 'restaurant', 'cafe'])
        
        # Skip test if no amenities found
        if amenities.empty:
            self.skipTest("No amenities found for testing")
        
        # Filter for accessible amenities
        accessible = self.downloader.filter_accessible_amenities(amenities)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(accessible, gpd.GeoDataFrame)
        
        # Check that it has the accessibility_score column
        self.assertIn('accessibility_score', accessible.columns)
        
        # Check that it has the accessibility_category column
        self.assertIn('accessibility_category', accessible.columns)
        
        # Check that it has the wheelchair_score column
        self.assertIn('wheelchair_score', accessible.columns)
    
    def test_filter_accessible_amenities_empty(self):
        """Test filtering accessible amenities with empty input."""
        # Create empty amenities GeoDataFrame
        empty_amenities = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Filter for accessible amenities
        accessible = self.downloader.filter_accessible_amenities(empty_amenities)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(accessible, gpd.GeoDataFrame)
        self.assertTrue(accessible.empty)
    
    def test_calculate_sidewalk_coverage(self):
        """Test calculating sidewalk coverage."""
        # Get street network and sidewalk data
        G = self.downloader.get_street_network(network_type='drive')
        sidewalks = self.downloader.get_sidewalk_data()
        
        # Calculate sidewalk coverage
        coverage = self.downloader.calculate_sidewalk_coverage(G, sidewalks)
        
        # Check that we got a dictionary
        self.assertIsInstance(coverage, dict)
        
        # Check that it has the expected keys
        expected_keys = [
            'road_length_km', 'sidewalk_length_km', 'sidewalk_coverage_ratio',
            'sidewalk_coverage_percent', 'crossing_count', 'curb_ramp_count',
            'intersection_count', 'area_km2', 'intersection_density',
            'crossing_density', 'crossings_per_intersection'
        ]
        for key in expected_keys:
            self.assertIn(key, coverage)
        
        # Check that the values are reasonable
        self.assertGreaterEqual(coverage['sidewalk_coverage_percent'], 0)
        self.assertLessEqual(coverage['sidewalk_coverage_percent'], 100)
        self.assertGreaterEqual(coverage['road_length_km'], 0)
        self.assertGreaterEqual(coverage['sidewalk_length_km'], 0)
    
    def test_analyze_amenity_accessibility(self):
        """Test analyzing amenity accessibility."""
        # Get amenities, sidewalks, and street network
        amenities = self.downloader.get_amenities(amenity_types=['school', 'restaurant', 'cafe'])
        sidewalks = self.downloader.get_sidewalk_data()
        G = self.downloader.get_street_network(network_type='walk')
        
        # Skip test if no amenities found
        if amenities.empty:
            self.skipTest("No amenities found for testing")
        
        # Analyze accessibility
        analyzed = self.downloader.analyze_amenity_accessibility(amenities, sidewalks, G)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(analyzed, gpd.GeoDataFrame)
        
        # Check that it has the expected columns
        expected_columns = [
            'has_nearby_sidewalk', 'distance_to_sidewalk',
            'has_nearby_crossing', 'distance_to_crossing',
            'walkability_score', 'walkability_category'
        ]
        for col in expected_columns:
            self.assertIn(col, analyzed.columns)
        
        # Check that walkability scores are reasonable
        self.assertTrue(all(analyzed['walkability_score'] >= 0))
        self.assertTrue(all(analyzed['walkability_score'] <= 100))
    
    def test_analyze_amenity_accessibility_empty_sidewalks(self):
        """Test analyzing amenity accessibility with no sidewalks."""
        # Get amenities and street network
        amenities = self.downloader.get_amenities(amenity_types=['school', 'restaurant', 'cafe'])
        G = self.downloader.get_street_network(network_type='walk')
        
        # Skip test if no amenities found
        if amenities.empty:
            self.skipTest("No amenities found for testing")
        
        # Create empty sidewalks GeoDataFrame
        empty_sidewalks = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Analyze accessibility
        analyzed = self.downloader.analyze_amenity_accessibility(amenities, empty_sidewalks, G)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(analyzed, gpd.GeoDataFrame)
        
        # Check that all amenities have no nearby sidewalks
        self.assertTrue(all(~analyzed['has_nearby_sidewalk']))
    
    def test_split_polygon(self):
        """Test splitting a polygon into chunks."""
        # Create a moderately sized test polygon within Bellevue (about 2-3 km²)
        large_polygon = Polygon([
            (-122.198, 47.607),
            (-122.198, 47.618),
            (-122.187, 47.618),
            (-122.187, 47.607),
            (-122.198, 47.607)
        ])
        
        # Split the polygon
        chunks = self.downloader._split_polygon(large_polygon, max_area_km2=0.5)
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk is a valid geometry
        for chunk in chunks:
            self.assertTrue(chunk.is_valid)
    
    def test_split_polygon_small(self):
        """Test splitting a small polygon (should return original)."""
        # Create a small test polygon
        small_polygon = Polygon([
            (-122.30, 47.65),
            (-122.30, 47.66),
            (-122.29, 47.66),
            (-122.29, 47.65),
            (-122.30, 47.65)
        ])
        
        # Split the polygon with large max area
        chunks = self.downloader._split_polygon(small_polygon, max_area_km2=100)
        
        # Should return the original polygon
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], small_polygon)
    
    def test_download_by_chunks(self):
        """Test downloading data by chunks."""
        # Create a moderately sized test polygon within Bellevue (about 3-4 km²)
        # This is large enough to test chunking but small enough to be fast
        large_polygon = Polygon([
            (-122.200, 47.605),
            (-122.200, 47.620),
            (-122.185, 47.620),
            (-122.185, 47.605),
            (-122.200, 47.605)
        ])
        
        # Download amenities by chunks (just schools for faster test)
        amenities = self.downloader.download_by_chunks(
            boundary=large_polygon,
            function_name='get_amenities',
            max_area_km2=0.8,  # Smaller chunks for faster processing
            amenity_types=['school']
        )
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(amenities, gpd.GeoDataFrame)
    
    def test_download_by_chunks_invalid_function(self):
        """Test downloading by chunks with invalid function name."""
        # Create a test polygon
        test_polygon = Polygon([
            (-122.30, 47.65),
            (-122.30, 47.66),
            (-122.29, 47.66),
            (-122.29, 47.65),
            (-122.30, 47.65)
        ])
        
        # Should raise ValueError for invalid function name
        with self.assertRaises(ValueError):
            self.downloader.download_by_chunks(
                boundary=test_polygon,
                function_name='invalid_function',
                max_area_km2=1
            )
    
    def test_calculate_isochrones(self):
        """Test calculating isochrones from a point."""
        # Skip this test if pandana is not installed
        try:
            import pandana
        except ImportError:
            self.skipTest("Pandana not installed")
        
        # Use a point in Bellevue test area
        origin_point = Point(-122.192, 47.612)
        
        # Calculate isochrones for shorter times to speed up the test
        isochrones = self.downloader.calculate_isochrones(
            origin_point, 
            travel_times=[1, 2],  # 1 and 2 minute walks
            network_type='walk'
        )
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(isochrones, gpd.GeoDataFrame)
        
        # Check that it has the expected columns
        self.assertIn('time', isochrones.columns)
        self.assertIn('geometry', isochrones.columns)
        
        # Check that we got isochrones for each time
        self.assertEqual(len(isochrones), 2)
    
    def test_calculate_isochrones_with_tuple(self):
        """Test calculating isochrones with tuple coordinates."""
        # Skip this test if pandana is not installed
        try:
            import pandana
        except ImportError:
            self.skipTest("Pandana not installed")
        
        # Use tuple coordinates (lat, lon) in Bellevue area
        origin_point = (47.612, -122.192)
        
        # Calculate isochrones
        isochrones = self.downloader.calculate_isochrones(
            origin_point, 
            travel_times=[1],  # 1 minute walk
            network_type='walk'
        )
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(isochrones, gpd.GeoDataFrame)
        self.assertEqual(len(isochrones), 1)
 
    def test_identify_mobility_barriers(self):
        """Test identifying mobility barriers."""
        # Get street network and sidewalk data
        G = self.downloader.get_street_network(network_type='walk')
        sidewalks = self.downloader.get_sidewalk_data()
        
        # Identify mobility barriers
        barriers = self.downloader.identify_mobility_barriers(G, sidewalks)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(barriers, gpd.GeoDataFrame)
        
        # Check that it has the expected columns
        self.assertIn('barrier_type', barriers.columns)
        self.assertIn('geometry', barriers.columns)
        
        # Check that barrier types are valid
        if not barriers.empty:
            valid_types = ['missing_sidewalk', 'crossing_without_ramp', 'disconnected_sidewalk']
            for barrier_type in barriers['barrier_type'].unique():
                self.assertIn(barrier_type, valid_types)
    
    def test_identify_mobility_barriers_empty_sidewalks(self):
        """Test identifying mobility barriers with no sidewalks."""
        # Get street network
        G = self.downloader.get_street_network(network_type='walk')
        
        # Create empty sidewalks GeoDataFrame
        empty_sidewalks = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Identify mobility barriers
        barriers = self.downloader.identify_mobility_barriers(G, empty_sidewalks)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(barriers, gpd.GeoDataFrame)
    
    def test_assess_osm_data_quality(self):
        """Test assessing OSM data quality."""
        # Assess data quality
        quality = self.downloader.assess_osm_data_quality()
        
        # Check that we got a dictionary
        self.assertIsInstance(quality, dict)
        
        # Check that it has the expected keys
        expected_keys = [
            'area_km2', 'node_count', 'edge_count', 'node_density', 'edge_density',
            'sidewalk_count', 'crossing_count', 'sidewalk_density', 'crossing_density',
            'amenity_count', 'amenity_density', 'overall_quality_score',
            'node_density_score', 'sidewalk_density_score', 'crossing_density_score',
            'amenity_density_score', 'wheelchair_tagging_rate'
        ]
        for key in expected_keys:
            self.assertIn(key, quality)
        
        # Check that the overall quality score is between 0 and 100
        self.assertGreaterEqual(quality['overall_quality_score'], 0)
        self.assertLessEqual(quality['overall_quality_score'], 100)
        
        # Check that individual scores are between 0 and 100
        score_keys = ['node_density_score', 'sidewalk_density_score', 
                     'crossing_density_score', 'amenity_density_score']
        for key in score_keys:
            self.assertGreaterEqual(quality[key], 0)
            self.assertLessEqual(quality[key], 100)
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        # Create test data
        test_data = {'test': 'data', 'value': 42}
        
        # Save data
        output_path = self.downloader.save_processed_data(
            test_data, "test_data", Path(self.temp_cache_dir), "json"
        )
        
        # Check that file was created
        self.assertTrue(output_path.exists())
        
        # Check that it's a JSON file
        self.assertEqual(output_path.suffix, '.json')
    
    def test_save_processed_data_geojson(self):
        """Test saving GeoDataFrame as GeoJSON."""
        # Create test GeoDataFrame
        test_gdf = gpd.GeoDataFrame(
            {'name': ['test'], 'geometry': [Point(0, 0)]},
            crs="EPSG:4326"
        )
        
        # Save data
        output_path = self.downloader.save_processed_data(
            test_gdf, "test_geojson", Path(self.temp_cache_dir), "geojson"
        )
        
        # Check that file was created
        self.assertTrue(output_path.exists())
        
        # Check that it's a GeoJSON file
        self.assertEqual(output_path.suffix, '.geojson')
    
    def test_save_processed_data_csv(self):
        """Test saving DataFrame as CSV."""
        # Create test DataFrame
        test_df = pd.DataFrame({'name': ['test'], 'value': [42]})
        
        # Save data
        output_path = self.downloader.save_processed_data(
            test_df, "test_csv", Path(self.temp_cache_dir), "csv"
        )
        
        # Check that file was created
        self.assertTrue(output_path.exists())
        
        # Check that it's a CSV file
        self.assertEqual(output_path.suffix, '.csv')
    
    def test_save_processed_data_invalid_format(self):
        """Test saving data with invalid format."""
        # Create test data
        test_data = {'test': 'data'}
        
        # Should raise ValueError for invalid format
        with self.assertRaises(ValueError):
            self.downloader.save_processed_data(
                test_data, "test", Path(self.temp_cache_dir), "invalid_format"
            )
    
    def test_retry_download(self):
        """Test retry download functionality."""
        # Test with a function that always fails
        def failing_function():
            raise Exception("Test failure")
        
        # Should return None after max retries
        result = self.downloader._retry_download(failing_function, max_retries=2)
        self.assertIsNone(result)
    
    def test_get_place_boundary(self):
        """Test getting place boundary."""
        # Test with a valid place name
        boundary = self.downloader._get_place_boundary("Seattle, WA")
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(boundary, gpd.GeoDataFrame)
        self.assertFalse(boundary.empty)
    
    def test_get_place_boundary_invalid(self):
        """Test getting boundary for invalid place name."""
        # Should raise an exception for invalid place
        with self.assertRaises(Exception):
            self.downloader._get_place_boundary("InvalidPlaceName12345")

if __name__ == '__main__':
    unittest.main()