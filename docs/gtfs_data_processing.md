# GTFS Data Processing

This document describes the GTFS (General Transit Feed Specification) data processing workflow used in the Urban Mobility Analytics project.

## Overview

The GTFS Transit Data Processing Module provides functionality to:

1. **Download GTFS feeds** from multiple transit agencies
2. **Extract transit stops** with accessibility information
3. **Calculate service frequency** and headway metrics
4. **Convert to GeoDataFrame** for spatial analysis
5. **Validate wheelchair accessibility** attributes
6. **Handle multiple agencies** in one city
7. **Generate comprehensive metrics** for mobility analysis

## What is GTFS?

GTFS (General Transit Feed Specification) is a standardized format for public transportation schedules and associated geographic information. It consists of several CSV files in a ZIP package:

| File | Description | Required | Current Usage |
|------|-------------|----------|---------------|
| stops.txt | Transit stops with locations and attributes | Yes | ✅ Full processing |
| routes.txt | Transit routes | Yes | ✅ Route information |
| trips.txt | Trips for each route | Yes | ✅ Service frequency |
| stop_times.txt | Times vehicles arrive at stops | Yes | ✅ Headway calculation |
| calendar.txt | Service dates | Yes | ✅ Service validation |
| agency.txt | Transit agency information | Yes | ✅ Agency metadata |
| calendar_dates.txt | Service exceptions | No | ✅ Exception handling |
| shapes.txt | Route shapes | No | 🔄 Planned |
| frequencies.txt | Headway-based service information | No | 🔄 Planned |

## Current Implementation

### Supported Agencies

| Agency | City | Status | Stops | Accessibility Rate |
|--------|------|--------|-------|-------------------|
| King County Metro | Seattle | ✅ Active | 4,123 | 95.7% |
| Sound Transit | Seattle | ✅ Active | 2,713 | 95.4% |
| **Total** | **Seattle** | **✅ Active** | **6,836** | **95.6%** |

### Processing Workflow

1. **Download GTFS Feed**
   - Fetch the GTFS zip file from the transit agency's website
   - Extract the contents to a temporary directory
   - Validate file structure and required files

2. **Extract Transit Stops**
   - Parse stops.txt to get stop locations and attributes
   - Extract wheelchair accessibility information
   - Clean and standardize the data
   - Handle missing or invalid coordinates

3. **Calculate Service Frequency**
   - Parse trips.txt and stop_times.txt
   - Calculate how many trips serve each stop per day
   - Calculate trips per hour and average headway
   - Handle different service patterns (weekday, weekend, etc.)

4. **Convert to GeoDataFrame**
   - Create Point geometries from latitude and longitude
   - Set coordinate reference system to EPSG:4326
   - Create a GeoDataFrame for spatial analysis
   - Add agency information for multi-agency support

5. **Validate Accessibility Data**
   - Check for missing or invalid wheelchair accessibility values
   - Calculate percentage of stops with known accessibility information
   - Identify stops that are explicitly not wheelchair accessible
   - Generate accessibility statistics

6. **Handle Multiple Agencies**
   - Merge GTFS feeds from multiple agencies
   - Resolve ID conflicts by prefixing with agency name
   - Create a unified view of transit service in a city
   - Maintain agency attribution for analysis

## Accessibility Information Processing

### GTFS Accessibility Fields

GTFS includes wheelchair accessibility information in several files:

1. **stops.txt**: The `wheelchair_boarding` field indicates if a stop is accessible:
   - 0 or empty = No information
   - 1 = Some vehicles at this stop can be boarded by a rider in a wheelchair
   - 2 = Wheelchair boarding is not possible at this stop

2. **trips.txt**: The `wheelchair_accessible` field indicates if a trip is accessible:
   - 0 or empty = No information
   - 1 = Vehicle can accommodate at least one rider in a wheelchair
   - 2 = Vehicle cannot accommodate riders in wheelchairs

### Processing Logic

```python
def process_accessibility(stops_df, trips_df):
    """
    Process wheelchair accessibility information from GTFS data.
    """
    # Map wheelchair_boarding values to accessibility status
    accessibility_map = {
        0: 'unknown',
        1: 'yes',
        2: 'no'
    }
    
    # Process stop-level accessibility
    stops_df['wheelchair_accessible'] = stops_df['wheelchair_boarding'].map(accessibility_map)
    
    # Process trip-level accessibility
    trip_accessibility = trips_df.groupby('route_id')['wheelchair_accessible'].agg(
        lambda x: 'yes' if 1 in x.values else ('no' if 2 in x.values else 'unknown')
    )
    
    return stops_df, trip_accessibility
```

## Service Frequency Calculation

### Metrics Calculated

1. **Trips per Day**: Total number of trips serving each stop
2. **Trips per Hour**: Average trips per hour during service hours
3. **Average Headway**: Average time between consecutive trips

### Calculation Process

```python
def calculate_service_frequency(stop_times_df, trips_df):
    """
    Calculate service frequency metrics for each stop.
    """
    # Group by stop and count trips
    stop_frequency = stop_times_df.groupby('stop_id').agg({
        'trip_id': 'nunique',  # Unique trips per stop
        'arrival_time': 'count'  # Total arrivals
    })
    
    # Calculate average headway
    stop_times_sorted = stop_times_df.sort_values(['stop_id', 'arrival_time'])
    stop_times_sorted['next_arrival'] = stop_times_sorted.groupby('stop_id')['arrival_time'].shift(-1)
    stop_times_sorted['headway_minutes'] = (
        stop_times_sorted['next_arrival'] - stop_times_sorted['arrival_time']
    ).dt.total_seconds() / 60
    
    avg_headway = stop_times_sorted.groupby('stop_id')['headway_minutes'].mean()
    
    return stop_frequency, avg_headway
```

## Current Data Quality

### Accessibility Coverage

| Agency | Total Stops | Accessible | Not Accessible | Unknown | Coverage Rate |
|--------|-------------|------------|----------------|---------|---------------|
| King County Metro | 4,123 | 3,945 | 178 | 0 | 100% |
| Sound Transit | 2,713 | 2,588 | 49 | 76 | 97.2% |
| **Total** | **6,836** | **6,533** | **227** | **76** | **98.9%** |

### Service Frequency Metrics

| Agency | Avg Trips/Day | Avg Trips/Hour | Avg Headway (min) |
|--------|---------------|----------------|-------------------|
| King County Metro | 72 | 6.1 | 9.8 |
| Sound Transit | 58 | 6.0 | 10.0 |
| **Overall** | **67** | **6.1** | **9.9** |

## Usage Examples

### Basic Processing

```python
from src.data_acquisition.gtfs_processor import GTFSProcessor

# Initialize the GTFS processor for King County Metro
processor = GTFSProcessor(agency_name="king_county_metro")

# Download the feed
processor.download_gtfs_feed()

# Load the feed
feed = processor.load_gtfs_feed()

# Extract stops with accessibility information
stops = processor.extract_stops()

# Convert to GeoDataFrame
stops_gdf = processor.stops_to_geodataframe(stops)

# Calculate service frequency
frequency = processor.calculate_service_frequency()

# Validate accessibility data
is_valid, issues = processor.validate_accessibility_data(stops)
```

### Multi-Agency Processing

```python
from src.data_acquisition.gtfs_processor import GTFSProcessor

# Process multiple agencies
agencies = ["king_county_metro", "sound_transit"]
all_stops = []

for agency in agencies:
    processor = GTFSProcessor(agency_name=agency)
    processor.download_gtfs_feed()
    stops = processor.extract_stops()
    stops['agency_name'] = agency
    all_stops.append(stops)

# Combine all stops
combined_stops = pd.concat(all_stops, ignore_index=True)
```

### Accessibility Analysis

```python
# Analyze wheelchair accessibility
accessible_stops = stops_gdf[stops_gdf['wheelchair_accessible'] == 'yes']
not_accessible = stops_gdf[stops_gdf['wheelchair_accessible'] == 'no']
unknown = stops_gdf[stops_gdf['wheelchair_accessible'] == 'unknown']

print(f"Accessible stops: {len(accessible_stops)} ({len(accessible_stops)/len(stops_gdf)*100:.1f}%)")
print(f"Not accessible: {len(not_accessible)} ({len(not_accessible)/len(stops_gdf)*100:.1f}%)")
print(f"Unknown: {len(unknown)} ({len(unknown)/len(stops_gdf)*100:.1f}%)")
```

### Service Frequency Analysis

```python
# Analyze service frequency
high_frequency = stops_gdf[stops_gdf['trips_per_day'] >= 100]
medium_frequency = stops_gdf[(stops_gdf['trips_per_day'] >= 50) & (stops_gdf['trips_per_day'] < 100)]
low_frequency = stops_gdf[stops_gdf['trips_per_day'] < 50]

print(f"High frequency stops (100+ trips/day): {len(high_frequency)}")
print(f"Medium frequency stops (50-99 trips/day): {len(medium_frequency)}")
print(f"Low frequency stops (<50 trips/day): {len(low_frequency)}")
```

## Output Format

### GeoParquet Files

Processed GTFS data is saved in GeoParquet format with the following structure:

**File Pattern:** `transit_stops_YYYYMMDD_HHMMSS.geoparquet`

**Key Columns:**
- `stop_id`: Unique stop identifier
- `stop_name`: Human-readable stop name
- `stop_lat`, `stop_lon`: Geographic coordinates
- `wheelchair_accessible`: Accessibility status ('yes', 'no', 'unknown')
- `trips_per_day`: Daily service frequency
- `trips_per_hour`: Hourly service frequency
- `avg_headway_minutes`: Average time between trips
- `agency_name`: Transit agency name
- `geometry`: Point geometry (EPSG:4326)

### Metadata

Each processed file includes comprehensive metadata:
- **Processing timestamp**: When the data was processed
- **Source agency**: Original transit agency
- **Data quality metrics**: Completeness, validity, consistency
- **Accessibility statistics**: Coverage and distribution
- **Service frequency statistics**: Min, max, mean values

## Quality Assurance

### Validation Checks

1. **Coordinate Validation**: Ensure lat/lon are within valid ranges
2. **Accessibility Validation**: Check for valid accessibility values
3. **Service Frequency Validation**: Verify positive trip counts
4. **Spatial Validation**: Ensure geometries are valid
5. **Data Completeness**: Check for required fields

### Error Handling

- **Missing files**: Graceful handling of optional GTFS files
- **Invalid coordinates**: Filter out stops with invalid lat/lon
- **Encoding issues**: Handle various text encodings
- **Service exceptions**: Process calendar_dates.txt for service changes

## Future Enhancements

### Planned Features

1. **Route Analysis**: Process route shapes and patterns
2. **Real-time Integration**: Connect to real-time GTFS feeds
3. **Multi-city Support**: Extend to additional cities
4. **Advanced Metrics**: Calculate transfer opportunities, network connectivity
5. **Visualization**: Generate route maps and service diagrams

### Performance Optimizations

1. **Parallel Processing**: Process multiple agencies simultaneously
2. **Caching**: Cache downloaded feeds to reduce API calls
3. **Incremental Updates**: Only process changed data
4. **Memory Optimization**: Handle large GTFS feeds efficiently

## Troubleshooting

### Common Issues

1. **Download Failures**: Check agency website for feed updates
2. **Encoding Errors**: Handle various text encodings in GTFS files
3. **Memory Issues**: Process large feeds in chunks
4. **Coordinate Errors**: Validate and filter invalid coordinates

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check feed structure
processor = GTFSProcessor("king_county_metro")
feed = processor.load_gtfs_feed()
print(f"Available files: {list(feed.keys())}")

# Validate specific file
stops_df = feed['stops']
print(f"Stops shape: {stops_df.shape}")
print(f"Required columns: {set(['stop_id', 'stop_name', 'stop_lat', 'stop_lon']) - set(stops_df.columns)}")
```