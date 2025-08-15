# src/utils/config_manager.py
"""
Configuration management for the urban mobility project.

This module handles all the settings and paths we need across the project.
It can load config from files (JSON/YAML) and environment variables,
and provides easy access to paths, API keys, and city-specific settings.
"""

import os
import json
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ConfigManager:
    """Manages project configuration and paths."""
    
    def __init__(self, config_path=None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to config file (JSON or YAML)
        """
        # Set up default project paths
        self.project_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.project_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Default configuration
        self.config = {
            "paths": {
                "project_dir": str(self.project_dir),
                "data_dir": str(self.data_dir),
                "raw_dir": str(self.raw_dir),
                "interim_dir": str(self.interim_dir),
                "processed_dir": str(self.processed_dir)
            },
            "crs": {
                "default": "EPSG:4326",  # WGS84 for storage
                "analysis": "EPSG:3857"   # Web Mercator for calculations
            },
            "cities": {
                "seattle": {
                    "name": "Seattle",
                    "state": "WA",
                    "state_fips": "53",
                    "county": "King",
                    "county_fips": "033",
                    "bbox": [-122.459696, 47.481002, -122.224433, 47.734136]
                }
            },
            "processing": {
                "chunk_size_km": 5,
                "max_distance_m": 1000,
                "buffer_sizes": {
                    "transit_stop": 400,
                    "amenity": 200
                }
            },
            "validation": {
                "schemas": {},
                "value_rules": {}
            },
            "api_keys": {}
        }
        
        # Load config file if provided
        if config_path:
            self.load_config(config_path)
        
        # Load environment variables
        self._load_env_overrides()
        
        # Create data directories
        self._create_directories()
    
    def load_config(self, config_path):
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to JSON or YAML config file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return False
            
            # Load based on file extension
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            else:
                logger.error(f"Unsupported file format: {config_path.suffix}")
                return False
            
            # Merge loaded config with defaults
            self._merge_config(self.config, loaded_config)
            logger.info(f"Loaded config from {config_path}")
            
            # Update path objects
            self._update_paths()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def save_config(self, config_path):
        """
        Save current configuration to a file.
        
        Args:
            config_path: Path to save config file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            
            # Save based on file extension
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.error(f"Unsupported file format: {config_path.suffix}")
                return False
            
            logger.info(f"Saved config to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key, default=None):
        """
        Get a configuration value using dot notation.
        
        Examples:
            config.get('crs.default')  # Get default CRS
            config.get('cities.seattle.name')  # Get Seattle name
            config.get('nonexistent', 'default_value')  # With default
        
        Args:
            key: Config key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        
        # Handle simple keys
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value using dot notation.
        
        Examples:
            config.set('crs.default', 'EPSG:4326')
            config.set('api_keys.census', 'your_api_key')
        
        Args:
            key: Config key (supports dot notation for nested keys)
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                config = self.config
                
                # Navigate to the deepest dict, creating missing levels
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                
                # Set the value
                config[parts[-1]] = value
            else:
                # Set simple key
                self.config[key] = value
            
            # Update path objects if we changed paths
            if key.startswith('paths.'):
                self._update_paths()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config value: {e}")
            return False
    
    def get_path(self, path_name):
        """
        Get a path from configuration.
        
        Args:
            path_name: Name of the path (e.g., 'data_dir', 'raw_dir')
            
        Returns:
            Path object or None if not found
        """
        path_str = self.get(f"paths.{path_name}")
        return Path(path_str) if path_str else None
    
    def get_city_config(self, city_name):
        """
        Get configuration for a specific city.
        
        Args:
            city_name: Name of the city (case-insensitive)
            
        Returns:
            City configuration dict or None if not found
        """
        return self.get(f"cities.{city_name.lower()}")
    
    def _merge_config(self, base_config, new_config):
        """
        Merge new config into base config, preserving nested structure.
        
        Args:
            base_config: Base configuration dict
            new_config: New configuration to merge
        """
        for key, value in new_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                # Recursively merge nested dicts
                self._merge_config(base_config[key], value)
            else:
                # Replace or add the value
                base_config[key] = value
    
    def _update_paths(self):
        """Update path objects from configuration strings."""
        self.project_dir = Path(self.config['paths']['project_dir'])
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.raw_dir = Path(self.config['paths']['raw_dir'])
        self.interim_dir = Path(self.config['paths']['interim_dir'])
        self.processed_dir = Path(self.config['paths']['processed_dir'])
    
    def _create_directories(self):
        """Create data directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir
        ]
        
        for directory in directories:
            if not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
    
    def _load_env_overrides(self):
        """Load configuration from environment variables."""
        # API keys from environment
        api_keys = {
            'CENSUS_API_KEY': 'api_keys.census',
            'MAPBOX_TOKEN': 'api_keys.mapbox'
        }
        
        for env_var, config_key in api_keys.items():
            value = os.environ.get(env_var)
            if value:
                self.set(config_key, value)
                logger.info(f"Loaded {env_var} from environment")
        
        # Add more environment overrides here as needed