# File: docs/census_data_processing.md
# Census Data Processing

This document describes the census data processing workflow used in the Urban Mobility Analytics project.

## Overview

The Census Data Acquisition Module (`census_fetcher.py`) provides functionality to:

1. Fetch census tract boundaries from the U.S. Census Bureau's TIGER/Line shapefiles
2. Fetch demographic data from the American Community Survey (ACS) 5-year estimates
3. Merge boundaries with demographic data for geospatial analysis
4. Validate and cache census data for efficiency

## Data Sources

### Census Tract Boundaries
- Source: U.S. Census Bureau TIGER/Line Shapefiles
- URL: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- Format: Shapefile (.shp)
- Content: Geographic boundaries of census tracts

### Demographic Data
- Source: American Community Survey (ACS) 5-year estimates
- API: Census Data API
- Documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html
- Format: JSON (via API)
- Content: Demographic variables at the census tract level

## Key Demographic Variables

| Variable Code | Description | Table |
|---------------|-------------|-------|
| B01001_001E | Total population | Sex by Age |
| B01002_001E | Median age | Median Age by Sex |
| B19013_001E | Median household income | Median Household Income |
| B08301_010E | Public transit commuters | Means of Transportation to Work |
| B08301_019E | Walk commuters | Means of Transportation to Work |
| B08301_021E | Bicycle commuters | Means of Transportation to Work |
| B18101_* | Disability status by age and sex | Sex by Age by Disability Status |

## Processing Workflow

1. **Fetch Census Tract Boundaries**
   - Use manually downloaded TIGER/Line shapefiles for reliability
   - Download files from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
   - Place files in the `data/raw/census/tiger/` directory
   - Filter for the target county
   - Standardize column names and CRS

2. **Fetch Demographic Data**
   - Query the Census API for demographic variables
   - Process the response into a DataFrame
   - Create a geoid column for joining with boundaries

3. **Merge Data**
   - Join boundaries with demographic data on the geoid column
   - Calculate derived metrics (e.g., commuting percentages, disability rates)
   - Validate the merged data

4. **Data Validation**
   - Check for missing geometries
   - Check for missing demographic values
   - Identify unreasonable values (e.g., extremely high incomes)
   - Flag census tracts with zero population

5. **Caching Mechanism**
   - Cache downloaded data to avoid repeated API calls
   - Set expiration time for cached data
   - Provide option to force refresh

## Usage Example

```python
from src.data_acquisition.census_fetcher import CensusFetcher

# Initialize the Census fetcher
fetcher = CensusFetcher(year=2021)

# Get census tract boundaries for King County, WA
boundaries = fetcher.get_census_boundaries("WA", "King")

# Get demographic data
demographics = fetcher.get_demographic_data(state="WA", county="King")

# Merge boundaries with demographic data
merged_data = fetcher.merge_census_data(boundaries, demographics)

# Validate the data
is_valid, issues = fetcher.validate_census_data(merged_data)

# Save the processed data
output_file = fetcher.save_processed_data(merged_data, "WA", "King")


Extending to Other Cities
The Census Fetcher is designed to work with any U.S. county. To use it for a different city:
Determine the state and county name or FIPS codes
Pass these to the fetcher functions
The fetcher will automatically handle the conversion between names and FIPS codes
Example for Portland, OR (Multnomah County):
# Get census data for Portland (Multnomah County, OR)
portland_data = fetcher.merge_census_data(state="OR", county="Multnomah")


Caching Behavior
Census data is cached in the data/interim/census_cache directory
Cache files are named based on the state, county, and data type
Cache expires after 30 days by default
Use use_cache=False to force a refresh of the data


Error Handling
The module includes robust error handling for common issues:
API rate limits or timeouts
Network connectivity problems
Missing or invalid data
Backup methods for API failures


Adapting for Multiple Cities
The CensusFetcher class is designed to work with any U.S. county, making it easy to extend to other cities. Here's how you would adapt it:
Define target cities: You can define additional target cities in a configuration file or directly in your code:
TARGET_CITIES = {
    "Seattle": {"state": "WA", "county": "King"},
    "Portland": {"state": "OR", "county": "Multnomah"},
    "San Francisco": {"state": "CA", "county": "San Francisco"},
    # Add more cities as needed
}

Process multiple cities: You can loop through the target cities to process data for all of them:
for city_name, city_info in TARGET_CITIES.items():
    print(f"Processing {city_name}...")
    data = fetch_and_process_census_data(city_info["state"], city_info["county"])
Comparison analysis: You can load processed data for multiple cities and perform comparative analysis:
# Load processed data for multiple cities
seattle_data = gpd.read_file("data/processed/census/census_53_033_2021.geojson")
portland_data = gpd.read_file("data/processed/census/census_41_051_2021.geojson")
sf_data = gpd.read_file("data/processed/census/census_06_075_2021.geojson")

# Perform comparative analysis
city_summaries = {
    "Seattle": seattle_data["total_population"].sum(),
    "Portland": portland_data["total_population"].sum(),
    "San Francisco": sf_data["total_population"].sum()
}
The key is that our CensusFetcher class is designed to be city-agnostic, taking state and county as parameters rather than hardcoding values for a specific city.

