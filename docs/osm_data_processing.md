# OpenStreetMap Data Processing

This document describes the OpenStreetMap (OSM) data processing workflow used in the Urban Mobility Analytics project.

## Table of Contents

1. [Overview](#overview)
2. [What is OpenStreetMap?](#what-is-openstreetmap)
3. [Key Data Types](#key-data-types)
4. [Accessibility Information in OSM](#accessibility-information-in-osm)
5. [Core Functionality](#core-functionality)
6. [Processing Workflow](#processing-workflow)
7. [Usage Examples](#usage-examples)
8. [Advanced Features](#advanced-features)
9. [Data Quality & Performance](#data-quality--performance)
10. [Testing & Validation](#testing--validation)

## Overview

The OSM Street Network & Amenities Module (`osm_downloader.py`) provides comprehensive functionality to:

1. **Download street networks** within geographic boundaries
2. **Extract pedestrian infrastructure** (sidewalks, crossings, curb ramps)
3. **Extract points of interest** (amenities) by type
4. **Filter amenities** for accessibility attributes
5. **Calculate sidewalk coverage** statistics
6. **Analyze amenity accessibility** based on nearby infrastructure
7. **Handle large area downloads** through intelligent chunking
8. **Identify mobility barriers** in the network
9. **Calculate isochrones** for accessibility analysis
10. **Assess data quality** and completeness

## What is OpenStreetMap?

OpenStreetMap (OSM) is a collaborative project to create a free and editable map of the world. It contains detailed data on:

- **Road networks** (streets, highways, paths)
- **Buildings** and structures
- **Points of interest** (amenities, services)
- **Natural features** (parks, water bodies)
- **Transit infrastructure** (bus stops, train stations)
- **Accessibility features** (sidewalks, curb ramps, tactile paving)

OSM data is particularly valuable for urban mobility analysis because it includes detailed information about pedestrian infrastructure and accessibility attributes that are often missing from other data sources.

## Key Data Types

### Street Networks

Street networks are represented as **NetworkX MultiDiGraph** objects:

- **Nodes**: Represent intersections and street endpoints
- **Edges**: Represent street segments with attributes
- **Attributes**: Include street type, name, surface, lanes, speed limits, etc.
- **Network Types**: drive, walk, bike, all (configurable)

### Sidewalks and Crossings

Sidewalk data is represented as **GeoDataFrames** containing:

- **Sidewalk geometries**: LineString features for pedestrian paths
- **Crossing locations**: LineString features for street crossings
- **Curb ramp locations**: Point features for accessibility ramps
- **Infrastructure types**: Categorized as 'sidewalk', 'crossing', 'curb_ramp', or 'other'
- **Surface information**: Material types and conditions
- **Width data**: When available from OSM tags
- **Tactile paving**: Presence of accessibility features

### Amenities

Amenity data is represented as **GeoDataFrames** containing:

- **Point locations**: Geographic coordinates of services
- **Amenity types**: Schools, hospitals, grocery stores, restaurants, etc.
- **Names and addresses**: When available in OSM
- **Wheelchair accessibility**: Detailed accessibility information
- **Accessibility scores**: Calculated walkability metrics
- **Proximity analysis**: Distance to nearby infrastructure

## Accessibility Information in OSM

OSM includes comprehensive accessibility tagging:

### 1. Wheelchair Accessibility
- `wheelchair=yes/no/limited/designated`
- Indicates if a location is accessible to wheelchair users
- Used to calculate accessibility scores

### 2. Tactile Features
- `tactile_paving=yes/no`
- Indicates presence of tactile paving for visually impaired users
- Important for universal design assessment

### 3. Curb Information
- `kerb=raised/lowered/flush`
- Describes the height of curbs at crossings
- Critical for mobility barrier identification

### 4. Surface Information
- `surface=paved/unpaved/asphalt/concrete/etc.`
- Describes the surface quality and material
- Affects walkability and accessibility

## Core Functionality

### OSMDownloader Class

The main class provides these key methods:

#### Initialization
```python
# By place name
downloader = OSMDownloader(place_name="Seattle, Washington")

# By custom boundary
downloader = OSMDownloader(boundary=custom_polygon)

# With custom cache folder
downloader = OSMDownloader(place_name="Seattle", cache_folder="/path/to/cache")
```

#### Street Network Download
```python
# Get different network types
drive_network = downloader.get_street_network(network_type='drive')
walk_network = downloader.get_street_network(network_type='walk')
all_network = downloader.get_street_network(network_type='all')

# With caching (default: True)
network = downloader.get_street_network(cache=True)
```

#### Pedestrian Infrastructure
```python
# Get all sidewalk and crossing data
sidewalk_data = downloader.get_sidewalk_data()

# Data includes:
# - infrastructure_type: 'sidewalk', 'crossing', 'curb_ramp', 'other'
# - highway, footway, sidewalk, crossing tags
# - surface, width, tactile_paving information
```

#### Amenity Extraction
```python
# Get specific amenity types
schools = downloader.get_amenities(amenity_types=['school'])
hospitals = downloader.get_amenities(amenity_types=['hospital'])

# Get default amenity types (comprehensive list)
all_amenities = downloader.get_amenities()

# Default types include: school, hospital, clinic, doctors, pharmacy,
# supermarket, grocery, library, community_centre, restaurant, cafe,
# bank, post_office, bus_station
```

#### Accessibility Analysis
```python
# Filter for accessible amenities
accessible = downloader.filter_accessible_amenities(amenities)

# Results include:
# - accessibility_score: 0-5 scale
# - accessibility_category: not_accessible, unknown, partially_accessible, fully_accessible
# - wheelchair_score: 0-3 scale based on wheelchair tag
```

## Processing Workflow

### 1. Boundary Definition
```python
# Automatic boundary from place name
boundary = downloader._get_place_boundary("Seattle, Washington")

# Custom boundary polygon
custom_boundary = Polygon([(lon1, lat1), (lon2, lat2), ...])
```

### 2. Street Network Extraction
- Uses OSMnx library for efficient network download
- Supports multiple network types (drive, walk, bike, all)
- Implements topology simplification for analysis
- Caches results as GraphML files for performance

### 3. Sidewalk and Crossing Extraction
- Extracts using specific OSM tag combinations:
  - Sidewalks: `highway=footway` + `footway=sidewalk` OR `sidewalk=*`
  - Crossings: `highway=footway` + `footway=crossing` OR `crossing=*`
  - Curb ramps: `kerb=*` OR footways with `incline`
- Categorizes infrastructure types automatically
- Handles missing data gracefully

### 4. Amenity Extraction
- Downloads points of interest by amenity type
- Filters for valid geometries
- Extracts accessibility attributes
- Caches results as GeoJSON files

### 5. Sidewalk Coverage Analysis
```python
coverage = downloader.calculate_sidewalk_coverage(street_network, sidewalk_data)

# Returns comprehensive statistics:
# - road_length_km: Total length of roads that should have sidewalks
# - sidewalk_length_km: Actual sidewalk length
# - sidewalk_coverage_percent: Percentage coverage
# - crossing_count: Number of pedestrian crossings
# - curb_ramp_count: Number of curb ramps
# - intersection_count: Number of street intersections
# - intersection_density: Intersections per km²
# - crossing_density: Crossings per km²
# - crossings_per_intersection: Average crossings per intersection
```

### 6. Amenity Accessibility Analysis
```python
amenities_with_access = downloader.analyze_amenity_accessibility(
    amenities, sidewalk_data, street_network
)

# Adds accessibility metrics:
# - has_nearby_sidewalk: Boolean flag
# - distance_to_sidewalk: Distance in meters
# - has_nearby_crossing: Boolean flag
# - distance_to_crossing: Distance in meters
# - walkability_score: 0-100 score
# - walkability_category: poor/fair/good/excellent
```

### 7. Large Area Processing
- Automatically splits large areas into manageable chunks
- Processes each chunk separately
- Combines results efficiently
- Implements retry logic with exponential backoff

## Usage Examples

### Basic Usage
```python
from src.data_acquisition.osm_downloader import OSMDownloader

# Initialize for Seattle
downloader = OSMDownloader(place_name="Seattle, Washington")

# Get core data
street_network = downloader.get_street_network(network_type='all')
sidewalk_data = downloader.get_sidewalk_data()
amenities = downloader.get_amenities()

# Analyze accessibility
accessible_amenities = downloader.filter_accessible_amenities(amenities)
coverage = downloader.calculate_sidewalk_coverage(street_network, sidewalk_data)
amenities_with_access = downloader.analyze_amenity_accessibility(
    amenities, sidewalk_data, street_network
)

# Save results
downloader.save_processed_data(street_network, "street_network", file_format="graphml")
downloader.save_processed_data(sidewalk_data, "sidewalks", file_format="geojson")
downloader.save_processed_data(accessible_amenities, "amenities", file_format="geojson")
```

### Advanced Usage with Custom Boundaries
```python
from shapely.geometry import Polygon

# Define custom boundary
custom_boundary = Polygon([
    (-122.35, 47.60), (-122.35, 47.65),
    (-122.30, 47.65), (-122.30, 47.60),
    (-122.35, 47.60)
])

# Initialize with custom boundary
downloader = OSMDownloader(boundary=custom_boundary)

# Process data within custom area
network = downloader.get_street_network()
sidewalks = downloader.get_sidewalk_data()
```

### Large Area Processing
```python
# For large cities, use chunking
large_area_amenities = downloader.download_by_chunks(
    boundary=city_boundary,
    function_name='get_amenities',
    max_area_km2=10,  # Maximum 10 sq km per chunk
    amenity_types=['school', 'hospital', 'grocery']
)
```

## Advanced Features

### Mobility Barrier Identification
```python
barriers = downloader.identify_mobility_barriers(street_network, sidewalk_data)

# Identifies:
# - missing_sidewalk: Major roads without nearby sidewalks
# - crossing_without_ramp: Crossings without curb ramps
# - disconnected_sidewalk: Isolated sidewalk segments
```

### Isochrone Calculation
```python
# Calculate walking accessibility areas (requires pandana)
isochrones = downloader.calculate_isochrones(
    origin_point=Point(-122.33, 47.61),
    travel_times=[5, 10, 15],  # 5, 10, 15 minute walks
    network_type='walk'
)
```

### Data Quality Assessment
```python
quality = downloader.assess_osm_data_quality()

# Comprehensive quality metrics:
# - node_density: Street network density
# - sidewalk_density: Pedestrian infrastructure density
# - crossing_density: Crossing infrastructure density
# - amenity_density: Points of interest density
# - wheelchair_tagging_rate: Accessibility information completeness
# - overall_quality_score: 0-100 quality score
```

### Data Export
```python
# Save in various formats
downloader.save_processed_data(data, "filename", file_format="geojson")
downloader.save_processed_data(data, "filename", file_format="csv")
downloader.save_processed_data(data, "filename", file_format="json")
downloader.save_processed_data(data, "filename", file_format="graphml")
```

## Data Quality & Performance

### Caching Strategy
- **Street networks**: Cached as GraphML files
- **Sidewalk data**: Cached as GeoJSON files
- **Amenity data**: Cached as GeoJSON files
- **Cache location**: `data/interim/osm_cache/`
- **Cache bypass**: Use `cache=False` parameter

### Error Handling
- **API timeouts**: Automatic retry with exponential backoff
- **Rate limiting**: Respects OSM API limits
- **Network issues**: Graceful degradation
- **Missing data**: Handles incomplete OSM data
- **Invalid geometries**: Filters out problematic features

### Performance Optimizations
- **Spatial indexing**: Efficient proximity calculations
- **Chunking**: Breaks large areas into manageable pieces
- **Simplified geometries**: Reduces processing time
- **Parallel processing**: Available for chunked downloads
- **Memory management**: Efficient data structures

### Data Quality Considerations
- **Completeness**: Varies by location (urban > rural)
- **Accessibility tagging**: Not all features have accessibility info
- **Currency**: OSM data is constantly updated
- **Consistency**: Tagging practices vary by region
- **Validation**: Module handles missing/invalid data gracefully

## Testing & Validation

### Unit Tests
The module includes comprehensive unit tests covering:
- Initialization with different parameters
- Street network download and caching
- Sidewalk data extraction
- Amenity filtering and analysis
- Large area chunking
- Error handling and edge cases
- Data quality assessment

### Running Tests
```bash
# Run all OSM tests
python -m unittest tests/data_acquisition/test_osm_downloader.py

# Run specific test
python -m unittest tests.data_acquisition.test_osm_downloader.TestOSMDownloader.test_get_street_network
```

### Example Script
```bash
# Run the complete OSM data processing pipeline
python src/data_acquisition/fetch_osm_data.py
```

### Validation Checks
- **Data integrity**: Ensures downloaded data is valid
- **Geometry validation**: Checks for valid spatial features
- **Attribute completeness**: Validates required fields
- **Performance benchmarks**: Monitors processing times
- **Memory usage**: Tracks resource consumption

## File Structure

```
src/data_acquisition/
├── osm_downloader.py          # Main OSM processing class
├── fetch_osm_data.py          # Example usage script
└── ...

tests/data_acquisition/
├── test_osm_downloader.py     # Comprehensive unit tests
└── ...

data/
├── raw/osm/                   # Raw OSM data (if downloaded separately)
├── interim/osm_cache/         # Cached processed data
└── processed/osm/             # Final processed outputs
```

## Dependencies

### Required Libraries
- **osmnx**: OpenStreetMap data download and processing
- **geopandas**: Geospatial data manipulation
- **networkx**: Graph data structures
- **shapely**: Geometric operations
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Optional Libraries
- **pandana**: For isochrone calculations
- **lxml**: For GraphML export
- **matplotlib**: For visualizations

### Installation
```bash
pip install osmnx geopandas networkx shapely pandas numpy
pip install pandana lxml matplotlib  # Optional
```

This documentation provides a comprehensive overview of the OSM data processing module, including all technical details, usage examples, and best practices for urban mobility analysis.