#!/usr/bin/env python3
"""
Data Quality Assessment Module

This module provides comprehensive data quality assessment functions for
urban mobility analytics datasets.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataQualityAssessor:
    """Comprehensive data quality assessment for urban mobility datasets."""
    
    def __init__(self):
        """Initialize the data quality assessor."""
        self.quality_metrics = {}
    
    def assess_census_data_quality(self, census_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Assess quality of census data.
        
        Args:
            census_data: Census GeoDataFrame
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Assessing census data quality")
        
        metrics = {
            'dataset': 'census',
            'timestamp': datetime.now().isoformat(),
            'completeness': {},
            'accuracy': {},
            'consistency': {},
            'spatial_quality': {},
            'overall_score': 0.0
        }
        
        if census_data is None or census_data.empty:
            metrics['overall_score'] = 0.0
            return metrics
        
        # Completeness checks
        total_rows = len(census_data)
        metrics['completeness'] = {
            'total_rows': total_rows,
            'missing_values': {},
            'completeness_rate': {}
        }
        
        # Check for missing values in key columns
        key_columns = ['geoid', 'total_population', 'geometry']
        for col in key_columns:
            if col in census_data.columns:
                missing_count = census_data[col].isna().sum()
                completeness_rate = (total_rows - missing_count) / total_rows
                metrics['completeness']['missing_values'][col] = missing_count
                metrics['completeness']['completeness_rate'][col] = completeness_rate
        
        # Accuracy checks
        metrics['accuracy'] = {
            'population_range': {
                'min': int(census_data['total_population'].min()) if 'total_population' in census_data.columns else 0,
                'max': int(census_data['total_population'].max()) if 'total_population' in census_data.columns else 0,
                'mean': float(census_data['total_population'].mean()) if 'total_population' in census_data.columns else 0
            },
            'negative_population': int((census_data['total_population'] < 0).sum()) if 'total_population' in census_data.columns else 0
        }
        
        # Consistency checks
        metrics['consistency'] = {
            'duplicate_geoids': int(census_data['geoid'].duplicated().sum()) if 'geoid' in census_data.columns else 0,
            'unique_geoids': int(census_data['geoid'].nunique()) if 'geoid' in census_data.columns else 0
        }
        
        # Spatial quality checks
        metrics['spatial_quality'] = {
            'valid_geometries': int(census_data.geometry.is_valid.sum()),
            'invalid_geometries': int((~census_data.geometry.is_valid).sum()),
            'empty_geometries': int(census_data.geometry.is_empty.sum()),
            'geometry_types': census_data.geometry.geom_type.value_counts().to_dict()
        }
        
        # Calculate overall score
        scores = []
        
        # Completeness score (30% weight)
        if metrics['completeness']['completeness_rate']:
            avg_completeness = np.mean(list(metrics['completeness']['completeness_rate'].values()))
            scores.append(avg_completeness * 0.3)
        
        # Accuracy score (25% weight)
        accuracy_score = 1.0
        if metrics['accuracy']['negative_population'] > 0:
            accuracy_score -= 0.5
        scores.append(accuracy_score * 0.25)
        
        # Consistency score (20% weight)
        consistency_score = 1.0
        if metrics['consistency']['duplicate_geoids'] > 0:
            consistency_score -= 0.5
        scores.append(consistency_score * 0.2)
        
        # Spatial quality score (25% weight)
        spatial_score = metrics['spatial_quality']['valid_geometries'] / total_rows
        scores.append(spatial_score * 0.25)
        
        metrics['overall_score'] = sum(scores)
        
        return metrics
    
    def assess_transit_data_quality(self, transit_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Assess quality of transit data.
        
        Args:
            transit_data: Transit stops GeoDataFrame
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Assessing transit data quality")
        
        metrics = {
            'dataset': 'transit',
            'timestamp': datetime.now().isoformat(),
            'completeness': {},
            'accuracy': {},
            'consistency': {},
            'spatial_quality': {},
            'service_quality': {},
            'overall_score': 0.0
        }
        
        if transit_data is None or transit_data.empty:
            metrics['overall_score'] = 0.0
            return metrics
        
        # Completeness checks
        total_stops = len(transit_data)
        metrics['completeness'] = {
            'total_stops': total_stops,
            'missing_values': {},
            'completeness_rate': {}
        }
        
        # Check for missing values in key columns
        key_columns = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'geometry']
        for col in key_columns:
            if col in transit_data.columns:
                missing_count = transit_data[col].isna().sum()
                completeness_rate = (total_stops - missing_count) / total_stops
                metrics['completeness']['missing_values'][col] = missing_count
                metrics['completeness']['completeness_rate'][col] = completeness_rate
        
        # Accuracy checks
        metrics['accuracy'] = {
            'lat_range': {
                'min': float(transit_data['stop_lat'].min()) if 'stop_lat' in transit_data.columns else 0,
                'max': float(transit_data['stop_lat'].max()) if 'stop_lat' in transit_data.columns else 0
            },
            'lon_range': {
                'min': float(transit_data['stop_lon'].min()) if 'stop_lon' in transit_data.columns else 0,
                'max': float(transit_data['stop_lon'].max()) if 'stop_lon' in transit_data.columns else 0
            },
            'invalid_coordinates': 0
        }
        
        # Check for invalid coordinates
        if 'stop_lat' in transit_data.columns and 'stop_lon' in transit_data.columns:
            invalid_lat = ((transit_data['stop_lat'] < -90) | (transit_data['stop_lat'] > 90)).sum()
            invalid_lon = ((transit_data['stop_lon'] < -180) | (transit_data['stop_lon'] > 180)).sum()
            metrics['accuracy']['invalid_coordinates'] = int(invalid_lat + invalid_lon)
        
        # Consistency checks
        metrics['consistency'] = {
            'duplicate_stop_ids': int(transit_data['stop_id'].duplicated().sum()) if 'stop_id' in transit_data.columns else 0,
            'unique_stop_ids': int(transit_data['stop_id'].nunique()) if 'stop_id' in transit_data.columns else 0,
            'agencies': transit_data['agency_name'].value_counts().to_dict() if 'agency_name' in transit_data.columns else {}
        }
        
        # Spatial quality checks
        metrics['spatial_quality'] = {
            'valid_geometries': int(transit_data.geometry.is_valid.sum()),
            'invalid_geometries': int((~transit_data.geometry.is_valid).sum()),
            'empty_geometries': int(transit_data.geometry.is_empty.sum()),
            'geometry_types': transit_data.geometry.geom_type.value_counts().to_dict()
        }
        
        # Service quality checks
        metrics['service_quality'] = {
            'stops_with_service': 0,
            'avg_trips_per_day': 0.0,
            'avg_headway_minutes': 0.0,
            'wheelchair_accessible_rate': 0.0
        }
        
        if 'trips_per_day' in transit_data.columns:
            stops_with_service = (transit_data['trips_per_day'] > 0).sum()
            metrics['service_quality']['stops_with_service'] = int(stops_with_service)
            metrics['service_quality']['avg_trips_per_day'] = float(transit_data['trips_per_day'].mean())
        
        if 'avg_headway_minutes' in transit_data.columns:
            metrics['service_quality']['avg_headway_minutes'] = float(transit_data['avg_headway_minutes'].mean())
        
        if 'wheelchair_accessible' in transit_data.columns:
            accessible_stops = (transit_data['wheelchair_accessible'] == 'yes').sum()
            metrics['service_quality']['wheelchair_accessible_rate'] = accessible_stops / total_stops
        
        # Calculate overall score
        scores = []
        
        # Completeness score (25% weight)
        if metrics['completeness']['completeness_rate']:
            avg_completeness = np.mean(list(metrics['completeness']['completeness_rate'].values()))
            scores.append(avg_completeness * 0.25)
        
        # Accuracy score (20% weight)
        accuracy_score = 1.0
        if metrics['accuracy']['invalid_coordinates'] > 0:
            accuracy_score -= 0.5
        scores.append(accuracy_score * 0.2)
        
        # Consistency score (15% weight)
        consistency_score = 1.0
        if metrics['consistency']['duplicate_stop_ids'] > 0:
            consistency_score -= 0.5
        scores.append(consistency_score * 0.15)
        
        # Spatial quality score (20% weight)
        spatial_score = metrics['spatial_quality']['valid_geometries'] / total_stops
        scores.append(spatial_score * 0.2)
        
        # Service quality score (20% weight)
        service_score = metrics['service_quality']['stops_with_service'] / total_stops
        scores.append(service_score * 0.2)
        
        metrics['overall_score'] = sum(scores)
        
        return metrics
    
    def assess_osm_data_quality(self, osm_data, data_type: str) -> Dict[str, Any]:
        """
        Assess quality of OSM data.
        
        Args:
            osm_data: OSM GeoDataFrame or NetworkX graph for street networks
            data_type: Type of OSM data ('sidewalks', 'amenities', 'street_network')
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info(f"Assessing OSM {data_type} data quality")
        
        metrics = {
            'dataset': f'osm_{data_type}',
            'timestamp': datetime.now().isoformat(),
            'completeness': {},
            'accuracy': {},
            'consistency': {},
            'spatial_quality': {},
            'overall_score': 0.0
        }
        
        if osm_data is None:
            metrics['overall_score'] = 0.0
            return metrics
        
        # Handle different data types
        if data_type == 'street_network':
            # For NetworkX graph objects
            if hasattr(osm_data, 'nodes') and hasattr(osm_data, 'edges'):
                total_features = 1  # Treat as single network
                is_network = True
            else:
                metrics['overall_score'] = 0.0
                return metrics
        else:
            # For GeoDataFrame objects
            if hasattr(osm_data, 'empty') and osm_data.empty:
                metrics['overall_score'] = 0.0
                return metrics
            if hasattr(osm_data, '__len__'):
                total_features = len(osm_data)
            else:
                total_features = 1
            is_network = False
        metrics['completeness'] = {
            'total_features': total_features,
            'missing_values': {},
            'completeness_rate': {}
        }
        
        # Check for missing values in key columns
        if is_network:
            # For network objects
            metrics['completeness']['missing_values'] = {}
            metrics['completeness']['completeness_rate'] = {'network': 1.0}
        elif hasattr(osm_data, 'columns'):
            # For GeoDataFrame objects
            key_columns = ['geometry']
            if data_type == 'sidewalks':
                key_columns.extend(['highway', 'surface'])
            elif data_type == 'amenities':
                key_columns.extend(['amenity', 'name'])
            
            for col in key_columns:
                if col in osm_data.columns:
                    missing_count = osm_data[col].isna().sum()
                    completeness_rate = (total_features - missing_count) / total_features
                    metrics['completeness']['missing_values'][col] = missing_count
                    metrics['completeness']['completeness_rate'][col] = completeness_rate
        else:
            # Fallback for other objects
            metrics['completeness']['missing_values'] = {}
            metrics['completeness']['completeness_rate'] = {'unknown': 1.0}
        
        # Spatial quality checks
        if is_network:
            # For network objects
            metrics['spatial_quality'] = {
                'valid_geometries': 1,
                'invalid_geometries': 0,
                'empty_geometries': 0,
                'geometry_types': {'network': 1}
            }
        elif hasattr(osm_data, 'geometry'):
            # For GeoDataFrame objects with geometry
            metrics['spatial_quality'] = {
                'valid_geometries': int(osm_data.geometry.is_valid.sum()),
                'invalid_geometries': int((~osm_data.geometry.is_valid).sum()),
                'empty_geometries': int(osm_data.geometry.is_empty.sum()),
                'geometry_types': osm_data.geometry.geom_type.value_counts().to_dict()
            }
        else:
            # For other objects
            metrics['spatial_quality'] = {
                'valid_geometries': 1,
                'invalid_geometries': 0,
                'empty_geometries': 0,
                'geometry_types': {'unknown': 1}
            }
        
        # Data-specific checks
        if data_type == 'sidewalks' and hasattr(osm_data, 'geometry'):
            metrics['sidewalk_specific'] = {
                'total_length_km': float(osm_data.geometry.length.sum() / 1000) if 'geometry' in osm_data.columns else 0,
                'avg_length_m': float(osm_data.geometry.length.mean()) if 'geometry' in osm_data.columns else 0
            }
        elif data_type == 'amenities' and hasattr(osm_data, 'columns'):
            metrics['amenity_specific'] = {
                'unique_amenity_types': int(osm_data['amenity'].nunique()) if 'amenity' in osm_data.columns else 0,
                'amenity_type_distribution': osm_data['amenity'].value_counts().to_dict() if 'amenity' in osm_data.columns else {}
            }
        elif data_type == 'street_network' and is_network:
            # For network objects
            try:
                metrics['network_specific'] = {
                    'total_nodes': len(osm_data.nodes()),
                    'total_edges': len(osm_data.edges()),
                    'is_connected': osm_data.is_connected() if hasattr(osm_data, 'is_connected') else True
                }
            except Exception as e:
                logger.warning(f"Could not assess street network quality: {e}")
                metrics['network_specific'] = {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'is_connected': False
                }
        
        # Calculate overall score
        scores = []
        
        # Completeness score (40% weight)
        if metrics['completeness']['completeness_rate']:
            avg_completeness = np.mean(list(metrics['completeness']['completeness_rate'].values()))
            scores.append(avg_completeness * 0.4)
        
        # Spatial quality score (60% weight)
        if total_features > 0:
            spatial_score = metrics['spatial_quality']['valid_geometries'] / total_features
        else:
            spatial_score = 1.0  # Default for network objects
        scores.append(spatial_score * 0.6)
        
        metrics['overall_score'] = sum(scores)
        
        return metrics
    
    def assess_mobility_index_quality(self, mobility_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Assess quality of mobility index data.
        
        Args:
            mobility_data: Mobility index GeoDataFrame
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Assessing mobility index data quality")
        
        metrics = {
            'dataset': 'mobility_index',
            'timestamp': datetime.now().isoformat(),
            'completeness': {},
            'accuracy': {},
            'consistency': {},
            'spatial_quality': {},
            'index_quality': {},
            'overall_score': 0.0
        }
        
        if mobility_data is None or mobility_data.empty:
            metrics['overall_score'] = 0.0
            return metrics
        
        # Completeness checks
        total_tracts = len(mobility_data)
        metrics['completeness'] = {
            'total_tracts': total_tracts,
            'missing_values': {},
            'completeness_rate': {}
        }
        
        # Check for missing values in key columns
        key_columns = ['geoid', 'mobility_access_index', 'transit_access_score', 'sidewalk_quality_score', 'amenity_proximity_score']
        for col in key_columns:
            if col in mobility_data.columns:
                missing_count = mobility_data[col].isna().sum()
                completeness_rate = (total_tracts - missing_count) / total_tracts
                metrics['completeness']['missing_values'][col] = missing_count
                metrics['completeness']['completeness_rate'][col] = completeness_rate
        
        # Index quality checks
        metrics['index_quality'] = {
            'mai_range': {
                'min': float(mobility_data['mobility_access_index'].min()) if 'mobility_access_index' in mobility_data.columns else 0,
                'max': float(mobility_data['mobility_access_index'].max()) if 'mobility_access_index' in mobility_data.columns else 0,
                'mean': float(mobility_data['mobility_access_index'].mean()) if 'mobility_access_index' in mobility_data.columns else 0
            },
            'component_scores': {}
        }
        
        # Check component scores
        component_columns = ['transit_access_score', 'sidewalk_quality_score', 'amenity_proximity_score', 'street_connectivity_score']
        for col in component_columns:
            if col in mobility_data.columns:
                metrics['index_quality']['component_scores'][col] = {
                    'min': float(mobility_data[col].min()),
                    'max': float(mobility_data[col].max()),
                    'mean': float(mobility_data[col].mean())
                }
        
        # Consistency checks
        metrics['consistency'] = {
            'duplicate_geoids': int(mobility_data['geoid'].duplicated().sum()) if 'geoid' in mobility_data.columns else 0,
            'unique_geoids': int(mobility_data['geoid'].nunique()) if 'geoid' in mobility_data.columns else 0
        }
        
        # Spatial quality checks
        metrics['spatial_quality'] = {
            'valid_geometries': int(mobility_data.geometry.is_valid.sum()),
            'invalid_geometries': int((~mobility_data.geometry.is_valid).sum()),
            'empty_geometries': int(mobility_data.geometry.is_empty.sum()),
            'geometry_types': mobility_data.geometry.geom_type.value_counts().to_dict()
        }
        
        # Calculate overall score
        scores = []
        
        # Completeness score (30% weight)
        if metrics['completeness']['completeness_rate']:
            avg_completeness = np.mean(list(metrics['completeness']['completeness_rate'].values()))
            scores.append(avg_completeness * 0.3)
        
        # Index quality score (40% weight)
        index_score = 1.0
        if 'mobility_access_index' in mobility_data.columns:
            # Check if scores are in reasonable range (0-100)
            mai_scores = mobility_data['mobility_access_index']
            if (mai_scores < 0).any() or (mai_scores > 100).any():
                index_score -= 0.3
        scores.append(index_score * 0.4)
        
        # Consistency score (15% weight)
        consistency_score = 1.0
        if metrics['consistency']['duplicate_geoids'] > 0:
            consistency_score -= 0.5
        scores.append(consistency_score * 0.15)
        
        # Spatial quality score (15% weight)
        spatial_score = metrics['spatial_quality']['valid_geometries'] / total_tracts
        scores.append(spatial_score * 0.15)
        
        metrics['overall_score'] = sum(scores)
        
        return metrics
    
    def generate_quality_report(self, all_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Args:
            all_metrics: Dictionary of quality metrics for all datasets
            
        Returns:
            Comprehensive quality report
        """
        logger.info("Generating comprehensive quality report")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'datasets_assessed': list(all_metrics.keys()),
            'overall_quality_score': 0.0,
            'dataset_scores': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Calculate overall score
        scores = []
        for dataset_name, metrics in all_metrics.items():
            score = metrics.get('overall_score', 0.0)
            report['dataset_scores'][dataset_name] = score
            scores.append(score)
        
        if scores:
            report['overall_quality_score'] = np.mean(scores)
        
        # Identify critical issues and warnings
        for dataset_name, metrics in all_metrics.items():
            score = metrics.get('overall_score', 0.0)
            
            if score < 0.5:
                report['critical_issues'].append(f"{dataset_name}: Very low quality score ({score:.2f})")
            elif score < 0.7:
                report['warnings'].append(f"{dataset_name}: Low quality score ({score:.2f})")
            
            # Check for specific issues
            if 'completeness' in metrics:
                completeness_rates = metrics['completeness'].get('completeness_rate', {})
                for col, rate in completeness_rates.items():
                    if rate < 0.8:
                        report['warnings'].append(f"{dataset_name}.{col}: Low completeness ({rate:.2f})")
            
            if 'spatial_quality' in metrics:
                invalid_geoms = metrics['spatial_quality'].get('invalid_geometries', 0)
                if invalid_geoms > 0:
                    report['warnings'].append(f"{dataset_name}: {invalid_geoms} invalid geometries")
        
        # Generate recommendations
        if report['overall_quality_score'] < 0.7:
            report['recommendations'].append("Overall data quality is below acceptable threshold. Review data sources and processing.")
        
        if report['critical_issues']:
            report['recommendations'].append("Address critical issues before proceeding with analysis.")
        
        if report['warnings']:
            report['recommendations'].append("Review warnings to improve data quality.")
        
        return report
