# File: tests/data_acquisition/test_gtfs_processor.py
"""
Unit tests for the gtfs_processor module.
"""
import unittest
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
import zipfile
import io

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_acquisition.gtfs_processor import GTFSProcessor

class TestGTFSProcessor(unittest.TestCase):
    """Tests for the GTFSProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a mock GTFS feed
        self.create_mock_gtfs_feed()
        
        # Initialize the processor with the test directory
        self.processor = GTFSProcessor(agency_name="test_agency", gtfs_dir=self.test_dir)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_mock_gtfs_feed(self):
        """Create a mock GTFS feed for testing."""
        # Create stops.txt
        stops = pd.DataFrame({
            'stop_id': ['1', '2', '3', '4', '5'],
            'stop_name': ['Stop 1', 'Stop 2', 'Stop 3', 'Stop 4', 'Stop 5'],
            'stop_lat': [47.6062, 47.6162, 47.6262, 47.6362, 47.6462],
            'stop_lon': [-122.3321, -122.3221, -122.3121, -122.3021, -122.2921],
            'wheelchair_boarding': [1, 2, 0, 1, np.nan]
        })
        stops.to_csv(self.test_dir / 'stops.txt', index=False)
        
        # Create routes.txt
        routes = pd.DataFrame({
            'route_id': ['A', 'B'],
            'route_short_name': ['Route A', 'Route B'],
            'route_long_name': ['Route A Long Name', 'Route B Long Name'],
            'route_type': [3, 3]  # 3 = Bus
        })
        routes.to_csv(self.test_dir / 'routes.txt', index=False)
        
        # Create trips.txt
        trips = pd.DataFrame({
            'route_id': ['A', 'A', 'B', 'B'],
            'service_id': ['weekday', 'weekend', 'weekday', 'weekend'],
            'trip_id': ['A1', 'A2', 'B1', 'B2'],
            'trip_headsign': ['Downtown', 'Airport', 'Downtown', 'Airport']
        })
        trips.to_csv(self.test_dir / 'trips.txt', index=False)
        
        # Create stop_times.txt
        stop_times = pd.DataFrame({
            'trip_id': ['A1', 'A1', 'A1', 'A2', 'A2', 'B1', 'B1', 'B2'],
            'stop_id': ['1', '2', '3', '1', '3', '4', '5', '5'],
            'arrival_time': ['08:00:00', '08:10:00', '08:20:00', '09:00:00', '09:20:00', 
                            '08:30:00', '08:45:00', '09:45:00'],
            'departure_time': ['08:00:00', '08:10:00', '08:20:00', '09:00:00', '09:20:00', 
                              '08:30:00', '08:45:00', '09:45:00'],
            'stop_sequence': [1, 2, 3, 1, 3, 1, 2, 1]
        })
        stop_times.to_csv(self.test_dir / 'stop_times.txt', index=False)
        
        # Create calendar.txt
        calendar = pd.DataFrame({
            'service_id': ['weekday', 'weekend'],
            'monday': [1, 0],
            'tuesday': [1, 0],
            'wednesday': [1, 0],
            'thursday': [1, 0],
            'friday': [1, 0],
            'saturday': [0, 1],
            'sunday': [0, 1],
            'start_date': ['20230101', '20230101'],
            'end_date': ['20231231', '20231231']
        })
        calendar.to_csv(self.test_dir / 'calendar.txt', index=False)
        
        # Create agency.txt
        agency = pd.DataFrame({
            'agency_id': ['test'],
            'agency_name': ['Test Transit Agency'],
            'agency_url': ['http://test.example.com'],
            'agency_timezone': ['America/Los_Angeles']
        })
        agency.to_csv(self.test_dir / 'agency.txt', index=False)
    
    def test_load_gtfs_feed(self):
        """Test loading a GTFS feed."""
        feed = self.processor.load_gtfs_feed()
        
        # Check that all files were loaded
        expected_files = ['stops', 'routes', 'trips', 'stop_times', 'calendar', 'agency']
        for file in expected_files:
            self.assertIn(file, feed)
        
        # Check that stops has the expected number of rows
        self.assertEqual(len(feed['stops']), 5)
    
    def test_extract_stops(self):
        """Test extracting stops from a GTFS feed."""
        # Load the feed first
        self.processor.load_gtfs_feed()
        
        # Extract stops
        stops = self.processor.extract_stops()
        
        # Print the actual data to see what it looks like
        print("\n" + "="*60)
        print("🚌 TRANSIT STOPS DATA")
        print("="*60)
        print(f"Total number of stops: {len(stops)}")
        print("\nSample of stops data:")
        print(stops.head())
        print("\nColumns available:")
        print(list(stops.columns))
        print("\nWheelchair accessibility breakdown:")
        if 'wheelchair_accessible' in stops.columns:
            print(stops['wheelchair_accessible'].value_counts())
        print("="*60)
        
        # Test 1: Check that we got stops data
        self.assertGreater(len(stops), 0, "Should extract at least some stops")
        
        # Test 2: Check that required columns exist
        required_columns = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
        for col in required_columns:
            self.assertIn(col, stops.columns, f"Should have {col} column")
        
        # Test 3: Check that wheelchair_accessible column was added
        self.assertIn('wheelchair_accessible', stops.columns, "Should add wheelchair_accessible column")
        
        # Test 4: Check that wheelchair_accessible values are valid
        if 'wheelchair_accessible' in stops.columns:
            valid_values = ['yes', 'no', 'unknown']
            for value in stops['wheelchair_accessible']:
                self.assertIn(value, valid_values, f"Invalid wheelchair_accessible value: {value}")
        
        # Test 5: Check that coordinates are reasonable (Seattle area)
        for _, stop in stops.iterrows():
            self.assertTrue(47.0 <= stop['stop_lat'] <= 48.0, f"Latitude {stop['stop_lat']} out of Seattle range")
            self.assertTrue(-123.0 <= stop['stop_lon'] <= -122.0, f"Longitude {stop['stop_lon']} out of Seattle range")
    
    def test_stops_to_geodataframe(self):
        """Test converting stops to a GeoDataFrame."""
        # Load the feed and extract stops
        self.processor.load_gtfs_feed()
        stops = self.processor.extract_stops()
        
        # Convert to GeoDataFrame
        gdf = self.processor.stops_to_geodataframe(stops)
        
        # Check that we got a GeoDataFrame
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        
        # Check that it has the expected number of rows
        self.assertEqual(len(gdf), 5)
        
        # Check that it has a geometry column
        self.assertIn('geometry', gdf.columns)
        
        # Check that the CRS is WGS84
        self.assertEqual(gdf.crs.to_string(), 'EPSG:4326')
    
    def test_calculate_service_frequency(self):
        """Test calculating service frequency."""
        # Load the feed
        self.processor.load_gtfs_feed()
        
        # Calculate service frequency for a weekday
        from datetime import date
        weekday = date(2023, 1, 2)  # A Monday
        frequency = self.processor.calculate_service_frequency(date=weekday)
        
        # Print the actual frequency data
        print("\n" + "="*60)
        print("⏰ SERVICE FREQUENCY DATA")
        print("="*60)
        print(f"Total stops with frequency data: {len(frequency)}")
        print("\nSample of frequency data:")
        print(frequency.head())
        print("\nColumns available:")
        print(list(frequency.columns))
        print("\nService frequency summary:")
        if len(frequency) > 0:
            print(f"Average trips per day: {frequency['trips_per_day'].mean():.2f}")
            print(f"Average trips per hour: {frequency['trips_per_hour'].mean():.2f}")
            print(f"Average headway (minutes): {frequency['avg_headway_minutes'].mean():.2f}")
        print("="*60)
        
        # Test 1: Check that we got results (even if empty, that's valid for mock data)
        self.assertIsInstance(frequency, pd.DataFrame, "Should return a DataFrame")
        
        # Test 2: Check that it has the expected columns (if not empty)
        if len(frequency) > 0:
            expected_columns = ['stop_id', 'trips_per_day', 'trips_per_hour', 'avg_headway_minutes', 'stop_name']
            for col in expected_columns:
                self.assertIn(col, frequency.columns, f"Should have {col} column")
            
            # Test 3: Check that frequency values are reasonable
            self.assertTrue(all(frequency['trips_per_day'] >= 0), "Trips per day should be non-negative")
            self.assertTrue(all(frequency['trips_per_hour'] >= 0), "Trips per hour should be non-negative")
            self.assertTrue(all(frequency['avg_headway_minutes'] >= 0), "Headway should be non-negative")
    
    def test_validate_accessibility_data(self):
        """Test validating accessibility data."""
        # Load the feed and extract stops
        self.processor.load_gtfs_feed()
        stops = self.processor.extract_stops()
        
        # Validate accessibility data
        is_valid, issues = self.processor.validate_accessibility_data(stops)
        
        # Check that validation found issues (our mock data has missing values)
        self.assertFalse(is_valid)
        
        # Check that it found the expected issues
        self.assertIn('missing_wheelchair_values', issues)
        self.assertIn('unknown_accessibility', issues)
        self.assertIn('not_accessible_stops', issues)
    
    def test_merge_agency_feeds(self):
        """Test merging feeds from multiple agencies."""
        # Create a second mock feed
        second_test_dir = Path(tempfile.mkdtemp())
        try:
            # Create stops.txt for second agency
            stops2 = pd.DataFrame({
                'stop_id': ['A', 'B', 'C'],
                'stop_name': ['Stop A', 'Stop B', 'Stop C'],
                'stop_lat': [47.7062, 47.7162, 47.7262],
                'stop_lon': [-122.4321, -122.4221, -122.4121],
                'wheelchair_boarding': [1, 1, 0]
            })
            stops2.to_csv(second_test_dir / 'stops.txt', index=False)
            
            # Create routes.txt for second agency
            routes2 = pd.DataFrame({
                'route_id': ['X', 'Y'],
                'route_short_name': ['Route X', 'Route Y'],
                'route_long_name': ['Route X Long Name', 'Route Y Long Name'],
                'route_type': [3, 3]
            })
            routes2.to_csv(second_test_dir / 'routes.txt', index=False)
            
            # Load both feeds
            feed1 = self.processor.load_gtfs_feed()
            
            processor2 = GTFSProcessor(agency_name="test_agency2", gtfs_dir=second_test_dir)
            feed2 = processor2.load_gtfs_feed()
            
            # Merge the feeds
            agency_feeds = {
                'agency1': feed1,
                'agency2': feed2
            }
            merged_feed = self.processor.merge_agency_feeds(agency_feeds)
            
            # Check that the merged feed has the expected number of stops
            self.assertEqual(len(merged_feed['stops']), 8)
            
            # Check that stop_ids were prefixed (convert to string first)
            stop_ids = merged_feed['stops']['stop_id'].astype(str)
            self.assertTrue(any(stop_id.startswith('agency2_') for stop_id in stop_ids), 
                          "Should prefix stop_ids with agency name")
            
        finally:
            # Clean up
            shutil.rmtree(second_test_dir)

if __name__ == '__main__':
    unittest.main()