# src/utils/data_persistence.py
"""
Data storage and retrieval utilities for the urban mobility project.

This module handles saving and loading datasets with metadata tracking.
It supports multiple formats (CSV, Parquet, GeoParquet, GeoJSON, SQLite)
and automatically organizes files by category with version tracking.
"""

import os
import pandas as pd
import geopandas as gpd
import sqlite3
import json
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Import spatial utilities for CRS handling
from .spatial_utils import ensure_crs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPersistence:
    """Handles saving and loading datasets with metadata tracking."""
    
    def __init__(self, base_dir=None, metadata_file=None):
        """
        Initialize the data persistence manager.
        
        Args:
            base_dir: Directory for storing data (default: project/data)
            metadata_file: Path to metadata file (default: base_dir/metadata.json)
        """
        # Set up paths
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent / "data"
        self.metadata_file = Path(metadata_file) if metadata_file else self.base_dir / "metadata.json"
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def save_dataframe(self, df, name, category=None, format="parquet", crs=None, metadata=None):
        """
        Save a DataFrame or GeoDataFrame with metadata tracking.
        
        Args:
            df: DataFrame or GeoDataFrame to save
            name: Name for the dataset
            category: Category for organization (e.g., 'census', 'transit', 'osm')
            format: File format ('csv', 'parquet', 'geoparquet', 'geojson', 'sqlite')
            crs: Coordinate system for GeoDataFrames
            metadata: Additional metadata to store
            
        Returns:
            Path to saved file, or None if failed
        """
        # Check for empty data
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning(f"Cannot save empty DataFrame: {name}")
            return None
        
        # Determine if it's a GeoDataFrame
        is_geo = isinstance(df, gpd.GeoDataFrame)
        
        # Set default category
        category = category or "general"
        
        # Create category directory
        category_dir = self.base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{name}_{timestamp}"
        
        # Set CRS for GeoDataFrames if specified
        if is_geo and crs:
            df = ensure_crs(df, crs)
        
        try:
            # Save based on format
            file_path = self._save_by_format(df, category_dir, file_name, format, is_geo, name)
            
            if file_path:
                # Update metadata
                self._update_metadata(name, category, file_path, df, format, metadata)
                logger.info(f"Saved {name} to {file_path}")
                return file_path
            
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}")
            return None
    
    def load_dataframe(self, name=None, category=None, version=None, file_path=None):
        """
        Load a DataFrame or GeoDataFrame.
        
        Args:
            name: Dataset name (optional if file_path provided)
            category: Dataset category (optional)
            version: Version to load (default: latest)
            file_path: Direct path to file (overrides other params)
            
        Returns:
            DataFrame or GeoDataFrame, or None if failed
        """
        try:
            # Load directly if file path provided
            if file_path:
                return self._load_file(Path(file_path))
            
            # Otherwise look up in metadata
            if not name:
                logger.error("Must provide either name or file_path")
                return None
            
            # Find matching datasets
            matches = []
            for dataset_id, info in self.metadata.get('datasets', {}).items():
                if info['name'] == name:
                    if category and info['category'] != category:
                        continue
                    matches.append(info)
            
            if not matches:
                logger.error(f"Dataset not found: {name}")
                return None
            
            # Sort by timestamp (newest first)
            matches.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Get specified version or latest
            if version is not None:
                dataset = next((m for m in matches if m['version'] == version), None)
                if not dataset:
                    logger.error(f"Version {version} not found for {name}")
                    return None
            else:
                dataset = matches[0]  # Latest version
            
            # Load the file
            return self._load_file(Path(dataset['file_path']))
            
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            return None
    
    def list_datasets(self, category=None, name=None):
        """
        List available datasets.
        
        Args:
            category: Filter by category
            name: Filter by name
            
        Returns:
            List of dataset info dictionaries
        """
        datasets = []
        
        for dataset_id, info in self.metadata.get('datasets', {}).items():
            if category and info['category'] != category:
                continue
            if name and info['name'] != name:
                continue
            datasets.append(info)
        
        # Sort by timestamp (newest first)
        datasets.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return datasets
    
    def get_latest_version(self, name, category=None):
        """
        Get the latest version of a dataset.
        
        Args:
            name: Dataset name
            category: Dataset category (optional)
            
        Returns:
            Dataset info dict or None if not found
        """
        datasets = self.list_datasets(category=category, name=name)
        return datasets[0] if datasets else None
    
    def delete_dataset(self, dataset_id=None, file_path=None):
        """
        Delete a dataset and its metadata.
        
        Args:
            dataset_id: Dataset ID in metadata
            file_path: Path to dataset file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find dataset in metadata
            if dataset_id and dataset_id in self.metadata.get('datasets', {}):
                info = self.metadata['datasets'][dataset_id]
                file_path = info['file_path']
            elif file_path:
                # Find by file path
                for id, info in self.metadata.get('datasets', {}).items():
                    if info['file_path'] == str(file_path):
                        dataset_id = id
                        break
            else:
                logger.error("Must provide either dataset_id or file_path")
                return False
            
            # Delete the file
            if file_path:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted file: {path}")
            
            # Remove from metadata
            if dataset_id and dataset_id in self.metadata.get('datasets', {}):
                del self.metadata['datasets'][dataset_id]
                self._save_metadata()
                logger.info(f"Removed dataset {dataset_id} from metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return False
    
    def _save_by_format(self, df, category_dir, file_name, format, is_geo, name):
        """
        Save DataFrame in the specified format.
        
        Args:
            df: DataFrame to save
            category_dir: Directory to save in
            file_name: Base filename
            format: File format
            is_geo: Whether it's a GeoDataFrame
            name: Dataset name for error messages
            
        Returns:
            Path to saved file, or None if failed
        """
        format_lower = format.lower()
        
        if format_lower == 'csv':
            file_path = category_dir / f"{file_name}.csv"
            df.to_csv(file_path, index=False)
            
        elif format_lower == 'parquet':
            file_path = category_dir / f"{file_name}.parquet"
            df.to_parquet(file_path, index=False)
            
        elif format_lower == 'geoparquet' and is_geo:
            file_path = category_dir / f"{file_name}.geoparquet"
            df.to_parquet(file_path)
            
        elif format_lower == 'geojson' and is_geo:
            file_path = category_dir / f"{file_name}.geojson"
            df.to_file(file_path, driver='GeoJSON')
            
        elif format_lower == 'sqlite':
            file_path = category_dir / f"{file_name}.sqlite"
            self._save_to_sqlite(df, name, file_path, is_geo)
            
        else:
            # Fallback to appropriate format
            if is_geo:
                logger.warning(f"Format {format} not suitable for GeoDataFrame, using GeoParquet")
                file_path = category_dir / f"{file_name}.geoparquet"
                df.to_parquet(file_path)
            else:
                logger.warning(f"Unsupported format {format}, using Parquet")
                file_path = category_dir / f"{file_name}.parquet"
                df.to_parquet(file_path, index=False)
        
        return file_path
    
    def _load_file(self, file_path):
        """
        Load a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame or GeoDataFrame, or None if failed
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path)
            
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            
            elif suffix == '.geoparquet':
                return gpd.read_parquet(file_path)
            
            elif suffix == '.geojson':
                return gpd.read_file(file_path)
            
            elif suffix == '.sqlite':
                return self._load_sqlite(file_path)
            
            else:
                logger.error(f"Unsupported file format: {suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def _load_sqlite(self, file_path):
        """
        Load data from SQLite file.
        
        Args:
            file_path: Path to SQLite file
            
        Returns:
            DataFrame or GeoDataFrame
        """
        # Get table name from metadata
        table_name = None
        for _, info in self.metadata.get('datasets', {}).items():
            if info['file_path'] == str(file_path):
                table_name = info.get('table_name', info['name'])
                break
        
        if not table_name:
            table_name = file_path.stem
        
        # Check if spatialite is available
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT CheckSpatialiteVersion()")
            has_spatialite = True
        except:
            has_spatialite = False
        
        if has_spatialite:
            # Try to load as GeoDataFrame
            try:
                return gpd.read_file(f"sqlite:///{file_path}", layer=table_name)
            except Exception:
                # Fall back to regular DataFrame
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        else:
            # Load as regular DataFrame
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    def _save_to_sqlite(self, df, table_name, file_path, is_geo=False):
        """
        Save DataFrame to SQLite.
        
        Args:
            df: DataFrame to save
            table_name: Table name
            file_path: SQLite file path
            is_geo: Whether it's a GeoDataFrame
        """
        if is_geo:
            df.to_file(f"sqlite:///{file_path}", layer=table_name, driver="SQLite")
        else:
            conn = sqlite3.connect(file_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
    
    def _load_metadata(self):
        """
        Load metadata from file.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        # Return empty metadata structure
        return {
            'datasets': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            self.metadata['last_updated'] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {self.metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _update_metadata(self, name, category, file_path, df, format, custom_metadata=None):
        """
        Update metadata for a saved dataset.
        
        Args:
            name: Dataset name
            category: Dataset category
            file_path: Path to saved file
            df: DataFrame that was saved
            format: Format used
            custom_metadata: Additional metadata
        """
        # Generate unique ID
        dataset_id = str(uuid.uuid4())
        
        # Basic dataset info
        info = {
            'id': dataset_id,
            'name': name,
            'category': category,
            'file_path': str(file_path),
            'format': format,
            'timestamp': datetime.now().isoformat(),
            'row_count': int(len(df)),  # Ensure it's a regular int
            'column_count': int(len(df.columns)),  # Ensure it's a regular int
            'columns': list(df.columns),
            'version': self._get_next_version(name, category)
        }
        
        # Add GeoDataFrame specific info
        if isinstance(df, gpd.GeoDataFrame):
            info['is_geo'] = True
            info['crs'] = str(df.crs)
            # Convert numpy int64 to regular int for JSON serialization
            geometry_counts = df.geometry.geom_type.value_counts()
            info['geometry_type'] = {k: int(v) for k, v in geometry_counts.items()}
            
            # Add bounding box if available
            try:
                minx, miny, maxx, maxy = df.total_bounds
                info['bounds'] = [float(minx), float(miny), float(maxx), float(maxy)]
            except:
                pass
        else:
            info['is_geo'] = False
        
        # Add custom metadata
        if custom_metadata:
            info['custom'] = custom_metadata
        
        # Add to metadata
        if 'datasets' not in self.metadata:
            self.metadata['datasets'] = {}
        
        self.metadata['datasets'][dataset_id] = info
        
        # Save metadata
        self._save_metadata()
    
    def _get_next_version(self, name, category):
        """
        Get the next version number for a dataset.
        
        Args:
            name: Dataset name
            category: Dataset category
            
        Returns:
            Next version number
        """
        # Find all versions of this dataset
        versions = []
        for info in self.metadata.get('datasets', {}).values():
            if info['name'] == name and info['category'] == category:
                if 'version' in info:
                    versions.append(info['version'])
        
        # Return next version
        return max(versions) + 1 if versions else 1