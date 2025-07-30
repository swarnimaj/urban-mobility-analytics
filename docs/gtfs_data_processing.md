# File: docs/gtfs_data_processing.md
# GTFS Data Processing

This document describes the GTFS (General Transit Feed Specification) data processing workflow used in the Urban Mobility Analytics project.

## Overview

The GTFS Transit Data Processing Module (`gtfs_processor.py`) provides functionality to:

1. Download GTFS feeds from transit agencies
2. Extract transit stops with accessibility information
3. Convert stops to GeoDataFrame for spatial analysis
4. Calculate service frequency and other metrics
5. Validate wheelchair accessibility attributes
6. Handle multiple transit agencies in one city

## What is GTFS?

GTFS (General Transit Feed Specification) is a standardized format for public transportation schedules and associated geographic information. It consists of several CSV files in a ZIP package:

| File | Description |
|------|-------------|
| stops.txt | Transit stops with locations and attributes |
| routes.txt | Transit routes |
| trips.txt | Trips for each route |
| stop_times.txt | Times vehicles arrive at stops |
| calendar.txt | Service dates |
| agency.txt | Transit agency information |
| shapes.txt | Route shapes (optional) |
| frequencies.txt | Headway-based service information (optional) |

## Accessibility Information in GTFS

GTFS includes wheelchair accessibility information in several files:

1. **stops.txt**: The `wheelchair_boarding` field indicates if a stop is accessible:
   - 0 or empty = No information
   - 1 = Some vehicles at this stop can be boarded by a rider in a wheelchair
   - 2 = Wheelchair boarding is not possible at this stop

2. **trips.txt**: The `wheelchair_accessible` field indicates if a trip is accessible:
   - 0 or empty = No information
   - 1 = Vehicle can accommodate at least one rider in a wheelchair
   - 2 = Vehicle cannot accommodate riders in wheelchairs

## Processing Workflow

1. **Download GTFS Feed**
   - Fetch the GTFS zip file from the transit agency's website
   - Extract the contents to a directory

2. **Extract Transit Stops**
   - Parse stops.txt to get stop locations and attributes
   - Extract wheelchair accessibility information
   - Clean and standardize the data

3. **Convert to GeoDataFrame**
   - Create Point geometries from latitude and longitude
   - Create a GeoDataFrame for spatial analysis

4. **Calculate Service Frequency**
   - Parse trips.txt and stop_times.txt
   - Calculate how many trips serve each stop per day
   - Calculate trips per hour and average headway

5. **Validate Accessibility Data**
   - Check for missing or invalid wheelchair accessibility values
   - Calculate percentage of stops with known accessibility information
   - Identify stops that are explicitly not wheelchair accessible

6. **Handle Multiple Agencies**
   - Merge GTFS feeds from multiple agencies
   - Resolve ID conflicts by prefixing with agency name
   - Create a unified view of transit service in a city

## Usage Example

```python
from src.data_acquisition.gtfs_processor import GTFSProcessor

# Initialize the GTFS processor
processor = GTFSProcessor(agency_name="king_county_metro")

# Download the feed
processor.download_gtfs_feed()

# Load the feed
feed = processor.load_gtfs_feed()

# Extract stops
stops = processor.extract_stops()

# Convert to GeoDataFrame
stops_gdf = processor.stops_to_geodataframe(stops)

# Calculate service frequency
frequency = processor.calculate_service_frequency()

# Validate accessibility data
is_valid, issues = processor.validate_accessibility_data(stops)

# Save processed data
processor.save_processed_data(stops_gdf, "stops", file_format="geojson")
processor.save_processed_data(frequency, "frequency", file_format="csv")
Working with Multiple Agencies
# Define agencies for Seattle
agency_urls = {
    "king_county_metro": "https://kingcounty.gov/~/media/depts/metro/data/gtfs/current-gtfs-zip",
    "sound_transit": "https://www.soundtransit.org/help-contacts/business-information/open-transit-data-otd/downloads"
}

# Create a processor
processor = GTFSProcessor()

# Process multiple agencies
result = processor.process_multiple_agencies(agency_urls, "Seattle")

Limitations and Considerations
GTFS feeds are updated periodically by transit agencies, so data may become outdated
Wheelchair accessibility information is often incomplete or missing
Service frequency calculations are based on scheduled service, not real-time data
Some agencies require registration or have usage terms for their GTFS feeds