# Data Sources Documentation

This document describes the data sources used in the Urban Mobility Analytics project, their schemas, and how to access them.

## Overview

The project integrates data from multiple sources to create a comprehensive mobility equity analysis:

1. **Census API**: Demographic and commuting data
2. **GTFS Feeds**: Public transportation schedules and stops
3. **OpenStreetMap**: Infrastructure and amenity data
4. **Local Government Data**: Additional context and validation

## Census API

### Description
The U.S. Census Bureau provides demographic and economic data through their API. We use the American Community Survey (ACS) 5-year estimates to obtain demographic information at the census tract level.

### Access Method
- **API Key required**: Yes (register at https://api.census.gov/data/key_signup.html)
- **Access through**: `census` Python package
- **Documentation**: https://www.census.gov/data/developers/data-sets/acs-5year.html
- **Rate limits**: 500 requests per day (with API key)

### Key Variables
| Variable Code | Description | Category |
|---------------|-------------|----------|
| B01001_001E | Total population | Demographics |
| B01001_002E | Male population | Demographics |
| B01001_026E | Female population | Demographics |
| B01002_001E | Median age | Demographics |
| B19013_001E | Median household income | Economics |
| B25077_001E | Median home value | Housing |
| B08301_001E | Total commuters | Commuting |
| B08301_010E | Public transit commuters | Commuting |
| B08301_019E | Walk commuters | Commuting |
| B08301_021E | Bicycle commuters | Commuting |
| B18101_001E | Total disability status determined | Disability |
| B18101_003E | Male with disability under 5 | Disability |
| B18101_018E | Female with disability under 5 | Disability |

### Sample Usage
```python
from src.data_acquisition.data_sources import CensusDataSource

# Initialize Census data source
census = CensusDataSource()

# Get population data for King County, WA
pop_data = census.get_population_data("53", "033")

# Get demographic data for King County, WA
demo_data = census.get_demographic_data("53", "033")

# Get commuting data for King County, WA
commute_data = census.get_commuting_data("53", "033")
```

### Current Coverage
- **State**: Washington (53)
- **County**: King County (033)
- **Census Tracts**: 495 tracts
- **Population**: 2,240,876 people
- **Data Year**: 2022 (ACS 5-year estimates)

## GTFS (General Transit Feed Specification)

### Description
GTFS is a common format for public transportation schedules and associated geographic information. Transit agencies publish GTFS feeds that include information about routes, stops, schedules, and accessibility features.

### Access Method
- **Manual download** from transit agency websites
- **Registration required** for some agencies
- **Usage terms** may apply
- **Update frequency**: Varies by agency (daily to monthly)

### Target Cities GTFS Sources

| City | Agency | URL | Status |
|------|--------|-----|--------|
| Seattle | King County Metro | https://kingcounty.gov/depts/transportation/metro/travel-options/bus/app-center/developer-resources.aspx | ✅ Active |
| Seattle | Sound Transit | https://www.soundtransit.org/help-contacts/business-information/open-transit-data-otd | ✅ Active |
| Portland | TriMet | https://developer.trimet.org/GTFS.shtml | 🔄 Planned |
| San Francisco | SFMTA | https://www.sfmta.com/reports/gtfs-transit-data | 🔄 Planned |

### Key Files in GTFS Feed

| File | Description | Required |
|------|-------------|----------|
| agency.txt | Transit agency information | Yes |
| stops.txt | Transit stop locations and attributes | Yes |
| routes.txt | Transit routes | Yes |
| trips.txt | Trips for each route | Yes |
| stop_times.txt | Times that vehicles arrive at stops | Yes |
| calendar.txt | Service dates | Yes |
| calendar_dates.txt | Service exceptions | No |
| shapes.txt | Route shapes | No |
| frequencies.txt | Headway-based service information | No |

### Accessibility Information in GTFS

GTFS includes wheelchair accessibility information in several files:

1. **stops.txt**: The `wheelchair_boarding` field indicates if a stop is accessible:
   - 0 or empty = No information
   - 1 = Some vehicles at this stop can be boarded by a rider in a wheelchair
   - 2 = Wheelchair boarding is not possible at this stop

2. **trips.txt**: The `wheelchair_accessible` field indicates if a trip is accessible:
   - 0 or empty = No information
   - 1 = Vehicle can accommodate at least one rider in a wheelchair
   - 2 = Vehicle cannot accommodate riders in wheelchairs

### Current GTFS Data

| Agency | Stops | Accessible Stops | Accessibility Rate | Trips/Day | Last Updated |
|--------|-------|------------------|-------------------|-----------|--------------|
| King County Metro | 4,123 | 3,945 | 95.7% | 72 | 2025-08-26 |
| Sound Transit | 2,713 | 2,588 | 95.4% | 58 | 2025-08-26 |
| **Total** | **6,836** | **6,533** | **95.6%** | **67** | **2025-08-26** |

### Sample Usage
```python
from src.data_acquisition.data_sources import GTFSDataSource

# Initialize GTFS data source
gtfs = GTFSDataSource()

# Get GTFS feed information for Seattle
feed_info = gtfs.get_gtfs_feed_info("Seattle")

# Download and process GTFS data
gtfs.download_gtfs_feed("king_county_metro")
stops = gtfs.extract_stops()
```

## OpenStreetMap (OSM)

### Description
OpenStreetMap is a collaborative project to create a free editable map of the world. It contains data about roads, buildings, amenities, and other geographic features with detailed accessibility information.

### Access Method
- **No API key required**
- **Access through**: `osmnx` Python package
- **Documentation**: https://osmnx.readthedocs.io/
- **Rate limits**: None (but be respectful of servers)

### Key Features

| Feature Type | Description | Current Coverage |
|--------------|-------------|------------------|
| Street Network | Road networks for driving, walking, or cycling | 162,972 segments |
| Sidewalks | Pedestrian infrastructure and crossings | 162,972 segments |
| Amenities | Points of interest like schools, hospitals, shops | 3,004 amenities |
| Buildings | Building footprints and attributes | 🔄 Planned |
| Accessibility | Features like wheelchair access, tactile paving | ✅ Active |

### OSM Tags for Accessibility

| Tag | Description | Values |
|-----|-------------|--------|
| `wheelchair` | Wheelchair accessibility | yes, limited, no, unknown |
| `tactile_paving` | Tactile paving presence | yes, no |
| `kerb` | Curb type | raised, lowered, flush |
| `crossing` | Crossing type | marked, unmarked, traffic_signals |
| `lit` | Lighting presence | yes, no |
| `surface` | Surface type | asphalt, concrete, gravel, etc. |

### Current OSM Data Coverage

| Data Type | Count | Area Coverage | Last Updated |
|-----------|-------|---------------|--------------|
| Sidewalks | 162,972 segments | King County, WA | 2025-08-26 |
| Amenities | 3,004 points | King County, WA | 2025-08-26 |
| Street Network | 162,972 segments | King County, WA | 2025-08-26 |

### Sample Usage
```python
from src.data_acquisition.data_sources import OSMDataSource

# Initialize OSM data source
osm = OSMDataSource()

# Get sidewalk data for King County
sidewalks = osm.get_sidewalk_data(
    north=47.780576, south=47.08435,
    east=-121.065945, west=-122.541661
)

# Get amenity data for King County
amenities = osm.get_amenity_data(
    north=47.780576, south=47.08435,
    east=-121.065945, west=-122.541661
)
```

## Local Government Data

### Description
Additional data sources from local government agencies provide context and validation for the analysis.

### Current Sources

| Source | Data Type | Coverage | Status |
|--------|-----------|----------|--------|
| King County GIS | Administrative boundaries | King County, WA | ✅ Active |
| Seattle Open Data | Additional transit data | Seattle, WA | 🔄 Planned |
| Washington State DOT | Highway data | Washington State | 🔄 Planned |

### Sample Usage
```python
from src.data_acquisition.data_sources import LocalDataSource

# Initialize local data source
local = LocalDataSource()

# Get administrative boundaries
boundaries = local.get_administrative_boundaries("King County")
```

## Data Integration

### Spatial Joins
All data sources are integrated using spatial relationships:

1. **Census Tracts** serve as the base geographic unit
2. **Transit Stops** are joined using spatial containment
3. **OSM Features** are joined using spatial intersection
4. **Local Data** provides additional context and validation

### Coordinate Reference System
All spatial data is standardized to **EPSG:4326** (WGS84) for consistency.

### Data Quality
Each data source includes quality metrics:
- **Completeness**: Percentage of non-null values
- **Validity**: Data type and range validation
- **Consistency**: Cross-field validation
- **Spatial quality**: Geometry validation

## Data Processing Pipeline

### Automated Workflow
1. **Data Download**: Automated fetching from all sources
2. **Data Cleaning**: Standardization and validation
3. **Spatial Processing**: Coordinate system alignment and spatial joins
4. **Quality Assessment**: Automated quality checks
5. **Integration**: Combined mobility index calculation
6. **Output**: GeoParquet files with metadata and lineage tracking

### File Naming Convention
All processed files follow the pattern: `{dataset_type}_{YYYYMMDD}_{HHMMSS}.geoparquet`

### Latest File Detection
The system automatically detects and loads the most recent file for each dataset type.

## Usage Examples

### Complete Pipeline
```python
from src.utils.pipeline_runner import PipelineRunner

# Run complete pipeline for Seattle
PipelineRunner.run_pipeline("Seattle")
```

### Individual Data Sources
```python
# Load latest data
from src.visualization.app import load_transit_data, load_mobility_data

transit_data = load_transit_data()
mobility_data = load_mobility_data()

# Analyze accessibility
accessible_stops = transit_data[transit_data['wheelchair_accessible'] == 'yes']
print(f"Accessibility rate: {len(accessible_stops)/len(transit_data)*100:.1f}%")
```

## Data Updates

### Update Frequency
- **Census Data**: Annual (ACS 5-year estimates)
- **GTFS Data**: Daily to monthly (varies by agency)
- **OSM Data**: Continuous (real-time updates)
- **Local Data**: Varies by source

### Version Control
All data versions are tracked with timestamps and metadata for reproducibility.

## Troubleshooting

### Common Issues
1. **Census API Limits**: Use API key for higher limits
2. **GTFS Download Failures**: Check agency website for updates
3. **OSM Rate Limiting**: Implement delays between requests
4. **Spatial Join Errors**: Verify coordinate reference systems

### Support
For data source issues, check:
- Agency websites for GTFS updates
- Census API documentation for variable changes
- OSM documentation for tag updates