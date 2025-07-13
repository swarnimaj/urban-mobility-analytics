"""
Utility functions for the Urban Mobility Analytics project.
"""
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

def ensure_directory(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    
def save_dataframe(df, filename, directory=PROCESSED_DATA_DIR, file_format="csv"):
    """
    Save a DataFrame to disk in the specified format.
    
    Args:
        df: DataFrame or GeoDataFrame to save
        filename: Name of the file (without extension)
        directory: Directory to save to
        file_format: Format to save as (csv, geojson, parquet, geoparquet)
    """
    ensure_directory(directory)
    filepath = Path(directory) / f"{filename}.{file_format}"
    
    if file_format == "csv":
        df.to_csv(filepath, index=False)
    elif file_format == "geojson" and isinstance(df, gpd.GeoDataFrame):
        df.to_file(filepath, driver="GeoJSON")
    elif file_format == "parquet":
        df.to_parquet(filepath, index=False)
    elif file_format == "geoparquet" and isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Saved {filename}.{file_format} to {directory}")
    
def load_dataframe(filename, directory=PROCESSED_DATA_DIR, file_format="csv", geo=False):
    """
    Load a DataFrame from disk.
    
    Args:
        filename: Name of the file (without extension)
        directory: Directory to load from
        file_format: Format to load (csv, geojson, parquet, geoparquet)
        geo: Whether to load as GeoDataFrame
        
    Returns:
        DataFrame or GeoDataFrame
    """
    filepath = Path(directory) / f"{filename}.{file_format}"
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if file_format == "csv":
        df = pd.read_csv(filepath)
        if geo:
            df = gpd.GeoDataFrame(df)
    elif file_format == "geojson":
        df = gpd.read_file(filepath)
    elif file_format == "parquet":
        df = pd.read_parquet(filepath)
        if geo:
            df = gpd.GeoDataFrame(df)
    elif file_format == "geoparquet":
        df = gpd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return df