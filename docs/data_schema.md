# Data Schema Documentation

This document describes the data structure for the urban mobility project.

## Overview

The project combines data from multiple sources to analyze mobility equity:
- **Census data**: Demographic and commuting information
- **Transit data**: Public transportation stops and service frequency  
- **OpenStreetMap data**: Infrastructure like sidewalks, streets, and amenities
- **Integrated data**: Combined mobility accessibility index

These datasets are processed and combined to create a comprehensive mobility accessibility index with normalized 0-1 scoring.

## Data File Formats

All processed data is stored in **GeoParquet** format for efficient spatial data handling:
- **Compressed**: Smaller file sizes than traditional formats
- **Spatial indexing**: Fast spatial queries
- **Columnar storage**: Efficient for analytics
- **CRS support**: Proper coordinate reference system handling

## Census Data

### Census Tracts

**File Pattern:** `census_tracts_YYYYMMDD_HHMMSS.geoparquet`

Basic demographic and commuting data for each census tract:

| Column | Type | Description |
|--------|------|-------------|
| `geoid` | string | Census tract identifier (GEOID) |
| `name` | string | Census tract name |
| `total_population` | float | Total population |
| `male_population` | float | Male population |
| `female_population` | float | Female population |
| `median_age` | float | Median age |
| `median_household_income` | float | Median household income ($) |
| `median_home_value` | float | Median home value ($) |
| `total_commuters` | float | Total number of commuters |
| `public_transit_commuters` | float | Number of public transit commuters |
| `walk_commuters` | float | Number of walking commuters |
| `bicycle_commuters` | float | Number of bicycle commuters |
| `pct_public_transit` | float | Percentage of commuters using public transit |
| `pct_walk` | float | Percentage of commuters walking |
| `pct_bicycle` | float | Percentage of commuters using bicycles |
| `total_with_disability` | float | Total population with disabilities |
| `pct_with_disability` | float | Percentage of population with disabilities |
| `state` | string | State abbreviation |
| `county` | string | County name |
| `tract` | string | Tract identifier |
| `geometry` | geometry | Census tract polygon geometry (EPSG:4326) |

## Transit Data

### Transit Stops

**File Pattern:** `transit_stops_YYYYMMDD_HHMMSS.geoparquet`

Information about public transportation stops with service frequency metrics:

| Column | Type | Description |
|--------|------|-------------|
| `stop_id` | string | Stop identifier |
| `stop_name` | string | Stop name |
| `stop_lat` | float | Stop latitude |
| `stop_lon` | float | Stop longitude |
| `stop_code` | string | Stop code |
| `stop_desc` | string | Stop description |
| `zone_id` | string | Fare zone identifier |
| `stop_url` | string | Stop URL |
| `location_type` | string | Location type (0=stop, 1=station) |
| `parent_station` | string | Parent station ID |
| `wheelchair_boarding` | string | Wheelchair boarding status |
| `stop_timezone` | string | Stop timezone |
| `platform_code` | string | Platform code |
| `tts_stop_name` | string | Text-to-speech stop name |
| `wheelchair_accessible` | string | Wheelchair accessibility status ('yes', 'no', 'unknown') |
| `trips_per_day` | float | Number of trips serving this stop per day |
| `trips_per_hour` | float | Average number of trips per hour |
| `avg_headway_minutes` | float | Average time between trips (minutes) |
| `agency_name` | string | Transit agency name |
| `geometry` | geometry | Stop point geometry (EPSG:4326) |

## OpenStreetMap Data

### Sidewalks

**File Pattern:** `sidewalks_YYYYMMDD_HHMMSS.geoparquet`

Pedestrian infrastructure data extracted from OpenStreetMap:

| Column | Type | Description |
|--------|------|-------------|
| `element` | string | OSM element type |
| `id` | int64 | OSM element ID |
| `highway` | string | Highway type (footway, path, etc.) |
| `source` | string | Data source |
| `traffic_signals:direction` | string | Traffic signal direction |
| `traffic_signals` | string | Traffic signals presence |
| `button_operated` | string | Button-operated signals |
| `crossing` | string | Crossing type |
| `crossing:island` | string | Crossing island presence |
| `crossing:markings` | string | Crossing markings |
| `crossing:signals` | string | Crossing signals |
| `lit` | string | Lighting presence |
| `tactile_paving` | string | Tactile paving presence |
| `bicycle` | string | Bicycle access |
| `traffic_signals:sound` | string | Traffic signal sound |
| `traffic_signals:vibration` | string | Traffic signal vibration |
| `flashing_lights` | string | Flashing lights |
| `traffic_calming` | string | Traffic calming features |
| `direction` | string | Direction |
| `stop` | string | Stop sign |
| `geometry` | geometry | LineString geometry (EPSG:4326) |

### Amenities

**File Pattern:** `amenities_YYYYMMDD_HHMMSS.geoparquet`

Points of interest and essential services:

| Column | Type | Description |
|--------|------|-------------|
| `element` | string | OSM element type |
| `id` | int64 | OSM element ID |
| `amenity` | string | Amenity type (school, hospital, shop, etc.) |
| `name` | string | Amenity name |
| `shop` | string | Shop type |
| `healthcare` | string | Healthcare type |
| `leisure` | string | Leisure type |
| `tourism` | string | Tourism type |
| `office` | string | Office type |
| `geometry` | geometry | Point geometry (EPSG:4326) |

## Integrated Data

### Mobility Index

**File Pattern:** `mobility_index_YYYYMMDD_HHMMSS.geoparquet`

Combined mobility accessibility index with normalized scoring:

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `geoid` | string | Census tract identifier | - |
| `geometry` | geometry | Census tract polygon | - |
| `area_sqkm` | float | Tract area in square kilometers | >0 |
| `total_population` | float | Total population | ≥0 |
| `stop_count` | float | Number of transit stops in tract | ≥0 |
| `stops_per_sqkm` | float | Transit stops per square kilometer | ≥0 |
| `length_km` | float | Sidewalk length in kilometers | ≥0 |
| `sidewalk_km_per_sqkm` | float | Sidewalk density per square kilometer | ≥0 |
| `amenity_count` | float | Number of amenities in tract | ≥0 |
| `amenities_per_sqkm` | float | Amenity density per square kilometer | ≥0 |
| `mobility_access_index` | float | **Normalized mobility accessibility index** | **0-1** |
| `transit_access_score` | float | Transit accessibility component | ≥0 |
| `sidewalk_quality_score` | float | Sidewalk quality component | ≥0 |
| `amenity_proximity_score` | float | Amenity proximity component | ≥0 |

### Sidewalk Scoring Fields (Sprint 8)

Additional fields added for comprehensive sidewalk and ramp accessibility scoring:

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `sidewalk_coverage_pct` | float | Sidewalk coverage percentage | 0-100 |
| `sidewalk_length_km` | float | Total sidewalk length in kilometers | ≥0 |
| `sidewalk_density_km_sqkm` | float | Sidewalk density per square kilometer | ≥0 |
| `total_crossings` | float | Number of pedestrian crossings | ≥0 |
| `crossings_with_ramps` | float | Crossings with curb ramps | ≥0 |
| `ramp_coverage_pct` | float | Curb ramp coverage percentage | 0-100 |
| `crossings_with_islands` | float | Crossings with pedestrian islands | ≥0 |
| `island_coverage_pct` | float | Pedestrian island coverage percentage | 0-100 |
| `coverage_score` | float | Normalized coverage score | 0-100 |
| `ramp_score` | float | Normalized ramp score | 0-100 |
| `island_score` | float | Normalized island score | 0-100 |
| `accessibility_score` | float | Combined accessibility score | 0-100 |
| `sidewalk_quality_score` | float | Final normalized sidewalk quality score | 0-100 |

### Amenity Proximity Fields (Sprint 9)

Additional fields added for comprehensive amenity proximity and accessibility scoring:

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `amenity_access_score` | float | Final normalized amenity access score | 0-100 |
| `amenity_density` | float | Amenity density per neighborhood | ≥0 |
| `amenity_diversity` | float | Number of unique amenity types | ≥0 |
| `raw_amenity_score` | float | Raw score before normalization | ≥0 |
| `avg_amenity_score` | float | Average score per amenity | ≥0 |
| `amenity_count` | int | Total number of amenities within range | ≥0 |
| `avg_distance` | float | Average distance to amenities (meters) | ≥0 |

## Metadata and Quality Tracking

### Metadata File

**File:** `metadata.json`

Comprehensive metadata about all datasets:

```json
{
  "datasets": {
    "uuid": {
      "id": "uuid",
      "name": "dataset_name",
      "category": "census|transit|osm|integrated",
      "file_path": "path/to/file.geoparquet",
      "format": "geoparquet",
      "timestamp": "ISO timestamp",
      "row_count": 1234,
      "column_count": 20,
      "columns": ["col1", "col2", ...],
      "version": 1,
      "is_geo": true,
      "crs": "EPSG:4326",
      "geometry_type": {"Point": 1000},
      "bounds": [min_lon, min_lat, max_lon, max_lat],
      "custom": {
        "validation": {...}
      }
    }
  },
  "last_updated": "ISO timestamp"
}
```

### Data Lineage

**File:** `data_lineage.json`

Complete tracking of data transformations:

```json
{
  "timestamp": "ISO timestamp",
  "sources": {
    "census": {...},
    "transit": {...},
    "osm": {...}
  },
  "transformations": [
    {
      "step": "data_preparation",
      "input": "raw_datasets",
      "output": "prepared_datasets",
      "description": "Prepared and standardized all datasets",
      "timestamp": "ISO timestamp"
    }
  ],
  "metadata": {
    "pipeline_version": "1.0.0",
    "run_timestamp": "ISO timestamp",
    "city": "Seattle",
    "total_sources": 3,
    "total_transformations": 6
  }
}
```

### Quality Report

**File:** `quality_report.json`

Data quality assessment results:

```json
{
  "overall_quality_score": 0.85,
  "datasets": {
    "census": {
      "completeness": 0.95,
      "validity": 0.90,
      "consistency": 0.88
    },
    "transit": {
      "completeness": 0.92,
      "validity": 0.94,
      "consistency": 0.91
    }
  },
  "timestamp": "ISO timestamp"
}
```

## Data Processing Pipeline

### File Naming Convention

All processed files follow the pattern: `{dataset_type}_{YYYYMMDD}_{HHMMSS}.geoparquet`

Examples:
- `census_tracts_20250826_234023.geoparquet`
- `transit_stops_20250826_225744.geoparquet`
- `sidewalks_20250826_234019.geoparquet`
- `mobility_index_20250826_234023.geoparquet`

### Latest File Detection

The system automatically detects and loads the most recent file for each dataset type based on the timestamp in the filename.

### Data Validation

Each dataset includes validation metadata:
- **Completeness**: Percentage of non-null values
- **Validity**: Data type and range validation
- **Consistency**: Cross-field validation
- **Spatial quality**: Geometry validation and CRS consistency

## Usage Examples

### Loading Data

```python
import geopandas as gpd
from pathlib import Path

# Load latest mobility index
mobility_files = list(Path("data/processed/integrated").glob("mobility_index_*.geoparquet"))
latest_file = max(mobility_files, key=lambda x: x.stat().st_mtime)
mobility_data = gpd.read_parquet(latest_file)

# Load latest transit data
transit_files = list(Path("data/processed/transit").glob("transit_stops_*.geoparquet"))
latest_file = max(transit_files, key=lambda x: x.stat().st_mtime)
transit_data = gpd.read_parquet(latest_file)
```

### Analyzing Mobility Index

```python
# Check mobility index distribution
print(f"Mobility index range: {mobility_data['mobility_access_index'].min():.3f} to {mobility_data['mobility_access_index'].max():.3f}")

# Find high-access tracts
high_access = mobility_data[mobility_data['mobility_access_index'] >= 0.7]
print(f"High access tracts: {len(high_access)}")

# Analyze by component
print(f"Average transit score: {mobility_data['transit_access_score'].mean():.1f}")
print(f"Average sidewalk score: {mobility_data['sidewalk_quality_score'].mean():.1f}")
print(f"Average amenity score: {mobility_data['amenity_proximity_score'].mean():.1f}")
```

### Analyzing Transit Accessibility

```python
# Check wheelchair accessibility
accessible = transit_data[transit_data['wheelchair_accessible'] == 'yes']
print(f"Accessibility rate: {len(accessible)/len(transit_data)*100:.1f}%")

# Analyze service frequency
print(f"Average trips per day: {transit_data['trips_per_day'].mean():.0f}")
print(f"Average headway: {transit_data['avg_headway_minutes'].mean():.1f} minutes")

# Agency analysis
agency_stats = transit_data.groupby('agency_name').agg({
    'stop_id': 'count',
    'wheelchair_accessible': lambda x: (x == 'yes').sum(),
    'trips_per_day': 'mean'
})
```