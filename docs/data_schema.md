# Data Schema Documentation

This document describes the data structure for the urban mobility project.

## Overview

The project combines data from three main sources to analyze mobility equity:
- **Census data**: Demographic and commuting information
- **Transit data**: Public transportation stops and service frequency  
- **OpenStreetMap data**: Infrastructure like sidewalks, streets, and amenities

These datasets are processed and combined to create a comprehensive mobility accessibility index.

## Census Data

### Census Tracts

**File:** `census_tracts.geoparquet`

Basic demographic and commuting data for each census tract:

| Column | Type | Description |
|--------|------|-------------|
| `geoid` | string | Census tract identifier (GEOID) |
| `name` | string | Census tract name |
| `total_population` | float | Total population |
| `median_household_income` | float | Median household income ($) |
| `total_commuters` | float | Total number of commuters |
| `public_transit_commuters` | float | Number of public transit commuters |
| `walk_commuters` | float | Number of walking commuters |
| `bicycle_commuters` | float | Number of bicycle commuters |
| `pct_public_transit` | float | Percentage of commuters using public transit |
| `pct_walk` | float | Percentage of commuters walking |
| `pct_bicycle` | float | Percentage of commuters using bicycles |
| `total_with_disability` | float | Total population with disabilities |
| `pct_with_disability` | float | Percentage of population with disabilities |
| `geometry` | geometry | Census tract polygon geometry |

## Transit Data

### Transit Stops

**File:** `transit_stops.geoparquet`

Information about public transportation stops:

| Column | Type | Description |
|--------|------|-------------|
| `stop_id` | string | Stop identifier |
| `stop_name` | string | Stop name |
| `wheelchair_accessible` | string | Wheelchair accessibility status ('yes', 'no', 'unknown') |
| `trips_per_day` | float | Number of trips serving this stop per day |
| `trips_per_hour` | float | Average number of trips per hour |
| `avg_headway_minutes` | float | Average time between trips (minutes) |
| `geometry` | geometry | Stop point geometry |

## OpenStreetMap Data

### Street Network

**File:** `street_network.graphml`

NetworkX graph representing the street network:

**Nodes:**
- `id`: Node identifier
- `x`: Longitude
- `y`: Latitude
- `street_count`: Number of streets connecting at this node

**Edges:**
- `osmid`: OSM way identifier
- `name`: Street name
- `highway`: Street type (residential, primary, etc.)
- `oneway`: Whether the street is one-way
- `length`: Length in meters
- `geometry`: LineString geometry of the street segment

### Sidewalks and Crossings

**File:** `sidewalks.geoparquet`

Pedestrian infrastructure data:

| Column | Type | Description |
|--------|------|-------------|
| `osmid` | int64 | OSM way identifier |
| `highway` | string | Highway type (footway, path, etc.) |
| `footway` | string | Footway type (sidewalk, crossing) |
| `surface` | string | Surface type (asphalt, concrete, etc.) |
| `width` | float | Width in meters (if available) |
| `tactile_paving` | string | Presence of tactile paving ('yes', 'no') |
| `kerb` | string | Curb type (raised, lowered, flush) |
| `infrastructure_type` | string | Categorized type (sidewalk, crossing, curb_ramp) |
| `length_km` | float | Length in kilometers |
| `geometry` | geometry | LineString geometry |

### Amenities

**File:** `amenities.geoparquet`

Points of interest with accessibility information:

| Column | Type | Description |
|--------|------|-------------|
| `osmid` | int64 | OSM node identifier |
| `amenity` | string | Amenity type (school, hospital, etc.) |
| `name` | string | Name of the amenity |
| `wheelchair` | string | Wheelchair accessibility ('yes', 'limited', 'no', 'unknown') |
| `wheelchair_score` | int | Numeric score for wheelchair accessibility (0-3) |
| `accessibility_score` | float | Overall accessibility score (0-100) |
| `accessibility_category` | string | Categorized accessibility (not_accessible, unknown, partially_accessible, fully_accessible) |
| `geometry` | geometry | Point geometry |

## Integrated Data

### Mobility Index

**File:** `mobility_index.geoparquet`

The main output dataset that combines all sources. Extends Census Tracts with mobility metrics:

| Column | Type | Description |
|--------|------|-------------|
| `stop_count` | int | Number of transit stops in the tract |
| `stops_per_sqkm` | float | Transit stops per square kilometer |
| `length_km` | float | Total sidewalk length in kilometers |
| `sidewalk_km_per_sqkm` | float | Sidewalk kilometers per square kilometer |
| `amenity_count` | int | Number of amenities in the tract |
| `accessible_amenity_count` | int | Number of accessible amenities |
| `pct_accessible_amenities` | float | Percentage of amenities that are accessible |
| `amenities_per_sqkm` | float | Amenities per square kilometer |
| `transit_access_score` | float | Transit access score (0-100) |
| `sidewalk_quality_score` | float | Sidewalk quality score (0-100) |
| `amenity_proximity_score` | float | Amenity proximity score (0-100) |
| `street_connectivity_score` | float | Street connectivity score (0-100) |
| `mobility_access_index` | float | Overall Mobility Accessibility Index (0-100) |
| `area_sqkm` | float | Area in square kilometers |
| `geometry` | geometry | Census tract polygon geometry |

## Data Relationships

The integrated data model connects datasets through spatial relationships:

1. **Census Tracts** serve as the base geographic unit
2. **Transit Stops** are joined to Census Tracts using spatial containment
3. **Sidewalks** are joined to Census Tracts using spatial intersection
4. **Amenities** are joined to Census Tracts using spatial containment
5. **Street Network** is analyzed within Census Tract boundaries

These relationships enable calculation of the Mobility Accessibility Index (MAI) at the census tract level, providing a comprehensive measure of mobility equity.

## Data Flow

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Census API   │    │  GTFS Feeds   │    │ OpenStreetMap │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Census Tracts │    │ Transit Stops │    │  OSM Features │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └─────────────┬──────┴─────────────┬──────┘
                      │                    │
                      ▼                    ▼
              ┌─────────────────┐    ┌────────────────┐
              │  Spatial Joins  │    │ Derived Metrics│
              └────────┬────────┘    └────────┬───────┘
                       │                      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │  Mobility Access   │
                       │      Index         │
                       └────────────────────┘
```

## Usage

The data processing pipeline automatically handles:
- Fetching data from all sources
- Standardizing coordinate systems
- Validating and cleaning data
- Performing spatial joins
- Calculating mobility metrics
- Creating visualizations

To run the pipeline for a city:

```python
from src.utils.pipeline_runner import PipelineRunner

runner = PipelineRunner()
results = runner.run_pipeline(
    city_name="Seattle",
    force_refresh=False
)
```

The pipeline creates all the datasets described above and saves them in the `data/processed/` directory.