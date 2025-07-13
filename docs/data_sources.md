# File: docs/data_sources.md
# Data Sources Documentation

This document describes the data sources used in the Urban Mobility Analytics project, their schemas, and how to access them.

## Census API

### Description
The U.S. Census Bureau provides demographic and economic data through their API. We use the American Community Survey (ACS) 5-year estimates to obtain demographic information at the county and census tract levels.

### Access Method
- API Key required: Yes (register at https://api.census.gov/data/key_signup.html)
- Access through: `census` Python package
- Documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html

### Key Variables
| Variable Code | Description |
|---------------|-------------|
| B01001_001E | Total population |
| B01002_001E | Median age |
| B19013_001E | Median household income |
| B08301_010E | Public transit commuters |
| B08301_019E | Walk commuters |
| B08301_021E | Bicycle commuters |

### Sample Usage
```python
from src.data_acquisition.data_sources import CensusDataSource

# Initialize Census data source
census = CensusDataSource()

# Get population data for King County, WA
pop_data = census.get_population_data("53", "033")

# Get demographic data for King County, WA
demo_data = census.get_demographic_data("53", "033")
GTFS (General Transit Feed Specification)
Description
GTFS is a common format for public transportation schedules and associated geographic information. Transit agencies publish GTFS feeds that include information about routes, stops, schedules, and more.

Access Method
Most transit agencies require manual download of GTFS feeds from their websites
Some agencies require registration or have usage terms
Target Cities GTFS Sources
City	Agency	URL
Seattle	King County Metro	https://kingcounty.gov/depts/transportation/metro/travel-options/bus/app-center/developer-resources.aspx
Portland	TriMet	https://developer.trimet.org/GTFS.shtml
San Francisco	SFMTA	https://www.sfmta.com/reports/gtfs-transit-data
Key Files in GTFS Feed
File	Description
agency.txt	Transit agency information
stops.txt	Transit stop locations
routes.txt	Transit routes
trips.txt	Trips for each route
stop_times.txt	Times that vehicles arrive at stops
calendar.txt	Service dates
Sample Usage
from src.data_acquisition.data_sources import GTFSDataSource

# Initialize GTFS data source
gtfs = GTFSDataSource()

# Get GTFS feed information for Seattle
feed_info = gtfs.get_gtfs_feed_info("Seattle")
OpenStreetMap (OSM)
Description
OpenStreetMap is a collaborative project to create a free editable map of the world. It contains data about roads, buildings, amenities, and other geographic features.

Access Method
No API key required
Access through: osmnx Python package
Documentation: https://osmnx.readthedocs.io/
Key Features
Feature Type	Description
Street Network	Road networks for driving, walking, or cycling
Amenities	Points of interest like schools, hospitals, shops
Buildings	Building footprints and attributes
Accessibility	Features like wheelchair access, tactile paving
Sample Usage
from src.data_acquisition.data_sources import OSMDataSource

# Initialize OSM data source
## OpenStreetMap (OSM)

### Description
OpenStreetMap is a collaborative project to create a free editable map of the world. It contains data about roads, buildings, amenities, and other geographic features.

### Access Method
- No API key required
- Access through: `osmnx` Python package
- Documentation: https://osmnx.readthedocs.io/

### Key Features
| Feature Type | Description |
|--------------|-------------|
| Street Network | Road networks for driving, walking, or cycling |
| Amenities | Points of interest like schools, hospitals, shops |
| Buildings | Building footprints and attributes |
| Accessibility | Features like wheelchair access, tactile paving |

### Sample Usage
```python
from src.data_acquisition.data_sources import OSMDataSource

# Initialize OSM data source
osm = OSMDataSource()

# Get street network for Seattle
graph = osm.get_street_network("Seattle, Washington", network_type="walk")

# Get amenities in Seattle
# First get the place boundary
place_name = "Seattle, WA, USA"
amenities = osm.get_amenities(place_name, ["pharmacy"])


Data Storage
Raw data is stored in the data/raw directory, organized by source:
data/raw/census/: Census data files
data/raw/gtfs/: GTFS feed files
data/raw/osm/: OpenStreetMap data files
Processed data is stored in the data/processed directory, organized by analysis type.