# Urban Mobility Analytics Platform

A geospatial analytics platform for visualizing urban mobility equity across neighborhoods, focusing on accessibility for people with disabilities, aging populations, and car-free commuters.

## Project Overview

This platform analyzes and visualizes how equitably urban infrastructure supports mobility across different neighborhoods by:
- Calculating multi-dimensional Mobility Accessibility Index (MAI)
- Visualizing transit access, sidewalk quality, amenity proximity, and street connectivity
- Identifying infrastructure gaps and equity concerns
- Providing actionable insights for urban planners and policymakers

## Features

- Interactive geospatial dashboard
- Multi-layer visualization of mobility metrics
- Neighborhood-level accessibility scoring
- Demographic and equity analysis overlays
- Natural language explanations of accessibility gaps

## Data Sources

- Census demographic data
- GTFS transit data
- OpenStreetMap infrastructure data
- Local government open datasets

## Getting Started

### Prerequisites
- Python 3.11
- pip

### Installation
1. Clone the repository
git clone https://github.com/swarnimaj/urban-mobility-analytics.git
cd urban-mobility-analytics

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run src/visualization/app.py

## Project Structure
urban-mobility-analytics/
├── data/ # Data files
│ ├── raw/ # Original, immutable data
│ ├── interim/ # Intermediate processed data
│ └── processed/ # Final, analysis-ready data
├── src/ # Source code
│ ├── data_acquisition/ # Data collection and processing
│ ├── analysis/ # Score calculation and analysis
│ └── visualization/ # Streamlit app and visualization
├── docs/ # Documentation
└── tests/ # Test cases
