"""
Configuration settings for the Urban Mobility Analytics project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project directory structure
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# API keys and credentials
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

# Default geographic settings
DEFAULT_CRS = "EPSG:4326"  # WGS84
ANALYSIS_CRS = "EPSG:3857"  # Web Mercator

# Default city for initial analysis
DEFAULT_CITY = "Seattle"
DEFAULT_STATE = "WA"

# Score weights
SCORE_WEIGHTS = {
    "transit_access": 0.25,
    "sidewalk_quality": 0.25,
    "amenity_proximity": 0.25,
    "street_connectivity": 0.25
}