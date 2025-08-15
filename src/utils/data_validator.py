# src/utils/data_validator.py
"""
Data validation utilities for checking data quality and structure.

This module helps ensure our data is clean and properly formatted before analysis.
It can check:
- Required columns and data types
- Value ranges and patterns
- Geometry validity for spatial data
- Relationships between datasets

The validator can load rules from config files or use built-in checks.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import spatial utilities for geometry validation
from .spatial_utils import validate_and_repair_geometries

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality and structure for geospatial datasets."""
    
    def __init__(self, validation_config=None):
        """
        Initialize the validator with optional configuration.
        
        Args:
            validation_config: Dict with validation rules or path to config file
        """
        self.validation_config = validation_config or {}
        self.validation_results = {}
        
        # Load config from file if path provided
        if isinstance(validation_config, str):
            try:
                with open(validation_config, 'r') as f:
                    self.validation_config = json.load(f)
                logger.info(f"Loaded validation config from {validation_config}")
            except Exception as e:
                logger.error(f"Failed to load validation config: {e}")
                self.validation_config = {}
    
    def validate_schema(self, df, schema_name=None):
        """
        Check if a DataFrame has the expected structure.
        
        Args:
            df: DataFrame or GeoDataFrame to check
            schema_name: Name of schema in config (optional)
            
        Returns:
            Dict with validation results
        """
        # Basic checks
        if df is None:
            return {'valid': False, 'errors': ['DataFrame is None']}
        
        if df.empty:
            return {'valid': False, 'errors': ['DataFrame is empty']}
        
        # Get schema from config if provided
        schema = None
        if schema_name and schema_name in self.validation_config.get('schemas', {}):
            schema = self.validation_config['schemas'][schema_name]
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        # Check required columns
        if schema and 'required_columns' in schema:
            missing_cols = [col for col in schema['required_columns'] if col not in df.columns]
            if missing_cols:
                results['errors'].extend([f"Missing required column: {col}" for col in missing_cols])
                results['valid'] = False
        
        # Check data types
        if schema and 'column_types' in schema:
            for col, expected_type in schema['column_types'].items():
                if col in df.columns:
                    actual_type = df[col].dtype.name
                    if not self._is_type_compatible(actual_type, expected_type):
                        results['errors'].append(f"Column {col} has type {actual_type}, expected {expected_type}")
                        results['valid'] = False
        
        # Check GeoDataFrame specifics
        if isinstance(df, gpd.GeoDataFrame):
            if 'geometry' not in df.columns:
                results['errors'].append("GeoDataFrame missing geometry column")
                results['valid'] = False
            elif df.geometry.isna().any():
                null_count = int(df.geometry.isna().sum())
                results['warnings'].append(f"Found {null_count} null geometries")
        
        return results
    
    def validate_values(self, df, rules=None, column_name=None):
        """
        Check data values against validation rules.
        
        Args:
            df: DataFrame to validate
            rules: Validation rules dict or rule set name
            column_name: Specific column to check (optional)
            
        Returns:
            Dict with validation results
        """
        if df is None or df.empty:
            return {'valid': False, 'errors': ['DataFrame is empty or None']}
        
        # Get rules from config if string provided
        if isinstance(rules, str) and rules in self.validation_config.get('value_rules', {}):
            rules = self.validation_config['value_rules'][rules]
        elif rules is None:
            rules = self.validation_config.get('value_rules', {})
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'columns_checked': []
        }
        
        # Determine which columns to check
        if column_name:
            if column_name not in df.columns:
                results['errors'].append(f"Column {column_name} not found")
                results['valid'] = False
                return results
            columns_to_check = [column_name]
        else:
            columns_to_check = [col for col in df.columns if col in rules]
        
        # Check each column
        for col in columns_to_check:
            results['columns_checked'].append(col)
            column_results = self._check_column_values(df, col, rules.get(col, {}))
            
            if not column_results['valid']:
                results['valid'] = False
                results['errors'].extend([f"{col}: {err}" for err in column_results['errors']])
                results['warnings'].extend([f"{col}: {warn}" for warn in column_results['warnings']])
        
        return results
    
    def validate_geometries(self, gdf):
        """
        Check geometry validity in a GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame to validate
            
        Returns:
            Dict with validation results
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            return {'valid': False, 'errors': ['Input is not a GeoDataFrame']}
        
        if gdf is None or gdf.empty:
            return {'valid': False, 'errors': ['GeoDataFrame is empty or None']}
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'geometry_types': {},
            'null_geometries': 0,
            'invalid_geometries': 0,
            'crs': str(gdf.crs) if gdf.crs else 'None'
        }
        
        # Check for null geometries
        null_count = int(gdf.geometry.isna().sum())
        results['null_geometries'] = null_count
        if null_count > 0:
            results['warnings'].append(f"Found {null_count} null geometries")
        
        # Check for invalid geometries
        non_null = gdf[~gdf.geometry.isna()]
        if not non_null.empty:
            invalid_count = int((~non_null.geometry.is_valid).sum())
            results['invalid_geometries'] = invalid_count
            if invalid_count > 0:
                results['errors'].append(f"Found {invalid_count} invalid geometries")
                results['valid'] = False
            
            # Count geometry types
            for geom_type in non_null.geometry.type.unique():
                count = int((non_null.geometry.type == geom_type).sum())
                results['geometry_types'][geom_type] = count
        
        # Check CRS
        if gdf.crs is None:
            results['warnings'].append("No CRS defined")
        
        return results
    
    def validate_dataset(self, dataset, dataset_name=None):
        """
        Run comprehensive validation on a dataset.
        
        Args:
            dataset: DataFrame or GeoDataFrame to validate
            dataset_name: Name for config lookup (optional)
            
        Returns:
            Dict with all validation results
        """
        results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Schema validation
        schema_results = self.validate_schema(dataset, dataset_name)
        if not schema_results['valid']:
            results['valid'] = False
            results['errors'].extend(schema_results['errors'])
            results['warnings'].extend(schema_results['warnings'])
        
        # Value validation (if rules exist)
        if dataset_name and dataset_name in self.validation_config.get('value_rules', {}):
            value_results = self.validate_values(dataset, dataset_name)
            if not value_results['valid']:
                results['valid'] = False
                results['errors'].extend(value_results['errors'])
                results['warnings'].extend(value_results['warnings'])
        
        # Geometry validation for GeoDataFrames
        if isinstance(dataset, gpd.GeoDataFrame):
            geom_results = self.validate_geometries(dataset)
            if not geom_results['valid']:
                results['valid'] = False
                results['errors'].extend(geom_results['errors'])
                results['warnings'].extend(geom_results['warnings'])
        
        # Store results
        self.validation_results[dataset_name] = results
        
        return results
    
    def generate_report(self, output_file=None):
        """
        Generate a simple validation report.
        
        Args:
            output_file: Path to save report (optional)
            
        Returns:
            Report as string
        """
        if not self.validation_results:
            return "No validation results to report"
        
        # Build report
        lines = [
            "# Data Validation Report",
            f"Generated: {datetime.now().isoformat()}",
            ""
        ]
        
        # Summary
        valid_count = sum(1 for r in self.validation_results.values() if r['valid'])
        total_count = len(self.validation_results)
        
        lines.extend([
            "## Summary",
            f"- Total datasets: {total_count}",
            f"- Valid: {valid_count}",
            f"- Invalid: {total_count - valid_count}",
            ""
        ])
        
        # Details for each dataset
        for name, results in self.validation_results.items():
            lines.extend([
                f"## {name}",
                f"Status: {'✅ Valid' if results['valid'] else '❌ Invalid'}",
                ""
            ])
            
            if results['errors']:
                lines.append("### Errors:")
                lines.extend([f"- {error}" for error in results['errors']])
                lines.append("")
            
            if results['warnings']:
                lines.append("### Warnings:")
                lines.extend([f"- {warning}" for warning in results['warnings']])
                lines.append("")
        
        report_text = "\n".join(lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_text
    
    def _check_column_values(self, df, column_name, rules):
        """
        Validate values in a single column.
        
        Args:
            df: DataFrame containing the column
            column_name: Column to validate
            rules: Validation rules for this column
            
        Returns:
            Dict with validation results
        """
        if column_name not in df.columns:
            return {'valid': False, 'errors': [f"Column {column_name} not found"]}
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        values = df[column_name]
        null_count = int(values.isna().sum())
        
        # Check null values
        if 'allow_nulls' in rules and not rules['allow_nulls'] and null_count > 0:
            results['errors'].append(f"Contains {null_count} null values but nulls not allowed")
            results['valid'] = False
        
        # Skip other checks if all values are null
        if null_count == len(df):
            results['warnings'].append("Column contains only null values")
            return results
        
        non_null = values.dropna()
        
        # Check data type
        if 'data_type' in rules:
            expected_type = rules['data_type']
            try:
                if expected_type == 'int':
                    pd.to_numeric(non_null, downcast='integer')
                elif expected_type == 'float':
                    pd.to_numeric(non_null)
                elif expected_type == 'bool':
                    non_null.astype(bool)
                elif expected_type == 'datetime':
                    pd.to_datetime(non_null)
                elif expected_type == 'string':
                    non_null.astype(str)
            except Exception:
                results['errors'].append(f"Values cannot be converted to {expected_type}")
                results['valid'] = False
        
        # Check numeric ranges
        if pd.api.types.is_numeric_dtype(non_null):
            min_val = non_null.min()
            max_val = non_null.max()
            
            if 'min_value' in rules and min_val < rules['min_value']:
                results['errors'].append(f"Minimum value {min_val} below allowed {rules['min_value']}")
                results['valid'] = False
            
            if 'max_value' in rules and max_val > rules['max_value']:
                results['errors'].append(f"Maximum value {max_val} above allowed {rules['max_value']}")
                results['valid'] = False
        
        # Check allowed values
        if 'allowed_values' in rules:
            invalid_values = set(non_null.unique()) - set(rules['allowed_values'])
            if invalid_values:
                results['errors'].append(f"Invalid values found: {invalid_values}")
                results['valid'] = False
        
        # Check string patterns
        if pd.api.types.is_string_dtype(non_null) or pd.api.types.is_object_dtype(non_null):
            if 'max_length' in rules:
                max_length = rules['max_length']
                str_values = non_null.astype(str)
                max_observed = str_values.str.len().max()
                
                if max_observed > max_length:
                    results['errors'].append(f"String length {max_observed} exceeds maximum {max_length}")
                    results['valid'] = False
            
            if 'pattern' in rules:
                pattern = rules['pattern']
                str_values = non_null.astype(str)
                match_count = int(str_values.str.match(pattern).sum())
                
                if match_count < len(non_null):
                    results['errors'].append(f"{len(non_null) - match_count} values don't match pattern")
                    results['valid'] = False
        
        return results
    
    def _is_type_compatible(self, actual_type, expected_type):
        """
        Check if data types are compatible.
        
        Args:
            actual_type: Actual pandas dtype
            expected_type: Expected type name
            
        Returns:
            Boolean indicating compatibility
        """
        # Exact match
        if actual_type == expected_type:
            return True
        
        # Numeric types
        if expected_type in ['int', 'integer', 'int64']:
            return actual_type in ['int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
        
        if expected_type in ['float', 'float64']:
            return actual_type in ['float', 'float16', 'float32', 'float64'] or self._is_type_compatible(actual_type, 'int')
        
        # String types
        if expected_type in ['str', 'string', 'text']:
            return actual_type in ['object', 'string', 'str']
        
        # Boolean types
        if expected_type in ['bool', 'boolean']:
            return actual_type in ['bool', 'boolean']
        
        # Datetime types
        if expected_type in ['datetime', 'date']:
            return actual_type in ['datetime64', 'datetime64[ns]', 'datetime64[ms]', 'datetime64[us]', 'datetime64[s]']
        
        return False