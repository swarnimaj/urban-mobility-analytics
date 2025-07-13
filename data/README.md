# Data Directory

This directory contains all data files used in the project.

## Structure

- `raw/`: Original, immutable data files downloaded from sources
- `interim/`: Intermediate data files that have been transformed or cleaned
- `processed/`: Final, analysis-ready data files used by the application

## Data Sources

- Census demographic data
- GTFS transit data
- OpenStreetMap infrastructure data
- Local government open datasets

Note: Large data files are not committed to the repository. The application will download or generate these files as needed.