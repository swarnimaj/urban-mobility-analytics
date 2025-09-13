# src/utils/spatial_utils.py
"""
Geospatial utility functions for urban mobility analysis.

This module handles common spatial operations we need for analyzing mobility data:
- Converting between coordinate systems (WGS84, Web Mercator, etc.)
- Joining datasets based on spatial relationships
- Calculating distances between points
- Creating buffer zones around features
- Fixing broken geometries
- Basic spatial analysis operations

Most functions include error handling and logging to help debug issues.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import wkt
import pyproj
from pathlib import Path
import logging
import warnings

# Set up logging - we'll use this to track what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common coordinate systems we'll use
WGS84 = "EPSG:4326"  # Standard GPS coordinates (lat/lon)
WEB_MERCATOR = "EPSG:3857"  # Web Mercator - good for area/distance calculations
NAD83 = "EPSG:4269"  # North American Datum 1983

# Defaults - WGS84 for storage, Web Mercator for calculations
DEFAULT_CRS = WGS84
DEFAULT_ANALYSIS_CRS = WEB_MERCATOR

def ensure_crs(gdf, target_crs=DEFAULT_CRS):
    """
    Make sure a GeoDataFrame has the right coordinate system.
    
    This is a common issue - sometimes data comes in without a CRS defined,
    or we need to convert between different coordinate systems for analysis.
    
    Args:
        gdf: The GeoDataFrame to fix
        target_crs: What coordinate system we want (default: WGS84)
        
    Returns:
        GeoDataFrame with the correct coordinate system
    """
    # Handle empty data gracefully
    if gdf is None or gdf.empty:
        return gdf
        
    try:
        # Case 1: No CRS defined at all
        if gdf.crs is None:
            logger.warning("GeoDataFrame missing CRS - setting to default")
            gdf.set_crs(target_crs, inplace=True)
            
        # Case 2: CRS exists but is different from what we want
        elif gdf.crs != target_crs:
            logger.info(f"Converting from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
            
        return gdf
        
    except Exception as e:
        logger.error(f"CRS conversion failed: {e}")
        return gdf

def safe_spatial_join(left_gdf, right_gdf, how="inner", predicate="intersects", **kwargs):
    """
    Join two GeoDataFrames based on spatial relationships, with error handling.
    
    Spatial joins can fail for various reasons - this function tries to handle
    the common issues gracefully.
    
    Args:
        left_gdf: Left side of the join
        right_gdf: Right side of the join  
        how: Join type ('inner', 'left', 'right')
        predicate: Spatial relationship ('intersects', 'contains', 'within', etc.)
        **kwargs: Extra arguments to pass to geopandas.sjoin
        
    Returns:
        Joined GeoDataFrame, or empty one if join fails
    """
    # Check for empty inputs - common source of errors
    if left_gdf is None or left_gdf.empty:
        logger.warning("Left GeoDataFrame is empty - can't join")
        return left_gdf
    
    if right_gdf is None or right_gdf.empty:
        logger.warning("Right GeoDataFrame is empty - can't join")
        # For left joins, return the left side; otherwise return empty
        return left_gdf if how == "left" else gpd.GeoDataFrame(geometry=[])
    
    # Make sure both datasets use the same coordinate system
    left_crs = left_gdf.crs
    right_gdf = ensure_crs(right_gdf, left_crs)
    
    try:
        logger.info(f"Joining datasets with {how} join and {predicate} relationship")
        joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate, **kwargs)
        logger.info(f"Join successful - {len(joined)} rows in result")
        return joined
        
    except Exception as e:
        logger.error(f"Spatial join failed: {e}")
        # Return appropriate fallback based on join type
        return left_gdf if how == "left" else gpd.GeoDataFrame(geometry=[], crs=left_crs)

def calculate_distances(origin_gdf, destination_gdf, max_distance=None, method='nearest'):
    """
    Find the distance from each origin point to destinations using various methods.
    
    This is useful for things like "how far is the nearest bus stop from each census tract?"
    We use a projected coordinate system for accurate distance calculations.
    
    Args:
        origin_gdf: Points we're measuring FROM
        destination_gdf: Points we're measuring TO
        max_distance: Optional maximum distance to consider (meters)
        method: Distance calculation method ('nearest', 'multiple', 'weighted')
        
    Returns:
        GeoDataFrame with distance metrics added
    """
    # Handle empty inputs
    if origin_gdf is None or origin_gdf.empty:
        logger.warning("No origin points provided")
        return origin_gdf
    
    if destination_gdf is None or destination_gdf.empty:
        logger.warning("No destination points provided")
        # Add infinite distance to indicate no destinations found
        result = origin_gdf.copy()
        result['nearest_dist'] = float('inf')
        return result
    
    # Convert to projected CRS for accurate distance calculations
    # Web Mercator is good for this in most cases
    analysis_crs = DEFAULT_ANALYSIS_CRS
    origin_proj = ensure_crs(origin_gdf.copy(), analysis_crs)
    dest_proj = ensure_crs(destination_gdf.copy(), analysis_crs)
    
    try:
        logger.info("Calculating distances to nearest destinations")
        
        # Set up result dataframe
        result = origin_gdf.copy()
        result['nearest_dist'] = float('inf')
        result['nearest_id'] = None
        
        # Calculate distances based on method
        if method == 'nearest':
            result = _calculate_nearest_distances(origin_proj, dest_proj, result, max_distance)
        elif method == 'multiple':
            result = _calculate_multiple_distances(origin_proj, dest_proj, result, max_distance)
        elif method == 'weighted':
            result = _calculate_weighted_distances(origin_proj, dest_proj, result, max_distance)
        else:
            logger.warning(f"Unknown method '{method}', using 'nearest'")
            result = _calculate_nearest_distances(origin_proj, dest_proj, result, max_distance)
        
        logger.info("Distance calculations complete")
        return result
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}")
        # Return original data with infinite distances
        result = origin_gdf.copy()
        result['nearest_dist'] = float('inf')
        return result

def create_buffers(gdf, distance, dissolve=False):
    """
    Create buffer zones around geometries.
    
    Useful for things like "what's within 400m of each transit stop?"
    
    Args:
        gdf: GeoDataFrame with geometries to buffer
        distance: Buffer distance in meters
        dissolve: Whether to merge all buffers into one polygon
        
    Returns:
        GeoDataFrame with buffer geometries
    """
    if gdf is None or gdf.empty:
        logger.warning("No geometries to buffer")
        return gdf
    
    # Convert to projected CRS for accurate buffer creation
    analysis_crs = DEFAULT_ANALYSIS_CRS
    gdf_proj = ensure_crs(gdf.copy(), analysis_crs)
    
    try:
        logger.info(f"Creating {distance}m buffers")
        
        # Create the buffers
        buffered = gdf_proj.copy()
        buffered['geometry'] = gdf_proj.geometry.buffer(distance)
        
        # Optionally merge all buffers into one
        if dissolve:
            logger.info("Merging all buffers into single polygon")
            buffered = buffered.dissolve()
            # Dissolve returns a Series, convert back to GeoDataFrame
            buffered = gpd.GeoDataFrame(geometry=[buffered.geometry.iloc[0]], crs=buffered.crs)
        
        # Convert back to original coordinate system
        buffered = ensure_crs(buffered, gdf.crs)
        
        logger.info(f"Buffer creation complete - {len(buffered)} geometries")
        return buffered
        
    except Exception as e:
        logger.error(f"Buffer creation failed: {e}")
        return gdf

def validate_and_repair_geometries(gdf):
    """
    Check for broken geometries and try to fix them.
    
    Invalid geometries can cause all sorts of problems with spatial operations.
    This function tries to repair common issues like self-intersections.
    
    Args:
        gdf: GeoDataFrame to check and repair
        
    Returns:
        GeoDataFrame with valid geometries
    """
    if gdf is None or gdf.empty:
        logger.warning("No geometries to validate")
        return gdf
    
    try:
        logger.info("Checking geometry validity")
        
        # Find invalid geometries
        invalid_mask = ~gdf.geometry.is_valid
        invalid_count = int(invalid_mask.sum())
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid geometries - attempting repair")
            
            # Make a copy so we don't modify the original
            repaired = gdf.copy()
            
            # Try to fix invalid geometries using buffer(0) trick
            # This often fixes self-intersections and other common issues
            repaired.loc[invalid_mask, 'geometry'] = repaired.loc[invalid_mask, 'geometry'].buffer(0)
            
            # Check if repair worked
            still_invalid = ~repaired.geometry.is_valid
            still_invalid_count = int(still_invalid.sum())
            
            if still_invalid_count > 0:
                logger.warning(f"{still_invalid_count} geometries couldn't be fixed - removing them")
                repaired = repaired[repaired.geometry.is_valid]
            
            logger.info("Geometry validation and repair complete")
            return repaired
        
        logger.info("All geometries are valid")
        return gdf
        
    except Exception as e:
        logger.error(f"Geometry validation failed: {e}")
        return gdf

def clip_to_boundary(gdf, boundary):
    """
    Clip a GeoDataFrame to a boundary polygon.
    
    Useful for limiting analysis to a specific area like a city boundary.
    
    Args:
        gdf: GeoDataFrame to clip
        boundary: GeoDataFrame or Polygon defining the boundary
        
    Returns:
        Clipped GeoDataFrame
    """
    if gdf is None or gdf.empty:
        logger.warning("No data to clip")
        return gdf
    
    # Handle different boundary input types
    if isinstance(boundary, gpd.GeoDataFrame):
        if boundary.empty:
            logger.warning("Boundary GeoDataFrame is empty")
            return gdf
        
        # If boundary has multiple polygons, merge them
        boundary_dissolved = boundary.dissolve()
        boundary_poly = boundary_dissolved.geometry.iloc[0]
    else:
        boundary_poly = boundary
    
    # Make sure boundary uses same CRS as data
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_poly], crs=gdf.crs)
    
    try:
        logger.info("Clipping data to boundary")
        clipped = gpd.clip(gdf, boundary_gdf)
        logger.info(f"Clipping complete - {len(clipped)} features remain")
        return clipped
        
    except Exception as e:
        logger.error(f"Clipping failed: {e}")
        return gdf

def calculate_area(gdf, area_column='area_sqkm'):
    """
    Calculate the area of polygon geometries in square kilometers.
    
    Args:
        gdf: GeoDataFrame with polygon geometries
        area_column: Name for the new area column
        
    Returns:
        GeoDataFrame with area column added
    """
    if gdf is None or gdf.empty:
        logger.warning("No polygons to calculate area for")
        return gdf
    
    # Convert to projected CRS for accurate area calculation
    analysis_crs = DEFAULT_ANALYSIS_CRS
    gdf_proj = ensure_crs(gdf.copy(), analysis_crs)
    
    try:
        logger.info("Calculating polygon areas")
        
        # Calculate area in square meters, then convert to square kilometers
        gdf_proj[area_column] = gdf_proj.geometry.area / 1e6
        
        # Convert back to original coordinate system
        result = ensure_crs(gdf_proj, gdf.crs)
        
        logger.info("Area calculation complete")
        return result
        
    except Exception as e:
        logger.error(f"Area calculation failed: {e}")
        return gdf

def calculate_length(gdf, length_column='length_km'):
    """
    Calculate the length of line geometries in kilometers.
    
    Args:
        gdf: GeoDataFrame with line geometries
        length_column: Name for the new length column
        
    Returns:
        GeoDataFrame with length column added
    """
    if gdf is None or gdf.empty:
        logger.warning("No lines to calculate length for")
        return gdf
    
    # Convert to projected CRS for accurate length calculation
    analysis_crs = DEFAULT_ANALYSIS_CRS
    gdf_proj = ensure_crs(gdf.copy(), analysis_crs)
    
    try:
        logger.info("Calculating line lengths")
        
        # Calculate length in meters, then convert to kilometers
        gdf_proj[length_column] = gdf_proj.geometry.length / 1000
        
        # Convert back to original coordinate system
        result = ensure_crs(gdf_proj, gdf.crs)
        
        logger.info("Length calculation complete")
        return result
        
    except Exception as e:
        logger.error(f"Length calculation failed: {e}")
        return gdf

def create_grid(boundary, cell_size_km=1.0):
    """
    Create a regular grid of cells covering a boundary.
    
    Useful for spatial analysis where you want uniform cells.
    
    Args:
        boundary: GeoDataFrame or Polygon defining the area to cover
        cell_size_km: Size of each grid cell in kilometers
        
    Returns:
        GeoDataFrame with grid cells
    """
    # Handle different boundary input types
    if isinstance(boundary, gpd.GeoDataFrame):
        if boundary.empty:
            logger.warning("Boundary is empty - can't create grid")
            return gpd.GeoDataFrame(geometry=[], crs=boundary.crs)
        
        crs = boundary.crs
        # Merge multiple polygons if present
        boundary_dissolved = boundary.dissolve()
        boundary_poly = boundary_dissolved.geometry.iloc[0]
    else:
        boundary_poly = boundary
        crs = DEFAULT_CRS
    
    # Convert to projected CRS for accurate grid creation
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_poly], crs=crs)
    boundary_proj = ensure_crs(boundary_gdf, DEFAULT_ANALYSIS_CRS)
    boundary_poly_proj = boundary_proj.geometry.iloc[0]
    
    try:
        # Get the bounding box
        minx, miny, maxx, maxy = boundary_poly_proj.bounds
        
        # Convert cell size to meters
        cell_size_m = cell_size_km * 1000
        
        logger.info(f"Creating {cell_size_km}km grid")
        
        # Calculate how many cells we need in each direction
        nx = int(np.ceil((maxx - minx) / cell_size_m))
        ny = int(np.ceil((maxy - miny) / cell_size_m))
        
        # Create all the grid cells
        grid_cells = []
        for i in range(nx):
            for j in range(ny):
                # Create a square cell
                cell = Polygon([
                    (minx + i * cell_size_m, miny + j * cell_size_m),
                    (minx + (i + 1) * cell_size_m, miny + j * cell_size_m),
                    (minx + (i + 1) * cell_size_m, miny + (j + 1) * cell_size_m),
                    (minx + i * cell_size_m, miny + (j + 1) * cell_size_m),
                    (minx + i * cell_size_m, miny + j * cell_size_m)  # Close the polygon
                ])
                grid_cells.append(cell)
        
        # Create GeoDataFrame from the cells
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=DEFAULT_ANALYSIS_CRS)
        
        # Add cell IDs for reference
        grid_gdf['cell_id'] = range(len(grid_gdf))
        
        # Clip to the boundary so we only keep cells that intersect
        grid_gdf = gpd.clip(grid_gdf, boundary_proj)
        
        # Convert back to original coordinate system
        grid_gdf = ensure_crs(grid_gdf, crs)
        
        logger.info(f"Grid creation complete - {len(grid_gdf)} cells")
        return grid_gdf
        
    except Exception as e:
        logger.error(f"Grid creation failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

def _calculate_nearest_distances(origin_proj, dest_proj, result, max_distance):
    """Calculate distance to nearest destination for each origin."""
    for idx, origin in origin_proj.iterrows():
        # Calculate distance to all destinations
        distances = dest_proj.geometry.distance(origin.geometry)
        
        # Find the minimum distance and its index
        if not distances.empty:
            min_idx = distances.idxmin()
            min_dist = distances.min()
            
            # Apply distance filter if specified
            if max_distance is not None and min_dist > max_distance:
                min_dist = float('inf')
                min_idx = None
        else:
            min_dist = float('inf')
            min_idx = None
        
        # Store the results
        result.loc[idx, 'nearest_dist'] = min_dist
        result.loc[idx, 'nearest_id'] = min_idx
    
    return result

def _calculate_multiple_distances(origin_proj, dest_proj, result, max_distance):
    """Calculate distances to multiple destinations within range."""
    # Add columns for multiple distance metrics
    result['nearest_dist'] = float('inf')
    result['nearest_id'] = None
    result['destinations_within_range'] = 0
    result['avg_dist_to_destinations'] = float('inf')
    
    for idx, origin in origin_proj.iterrows():
        # Calculate distance to all destinations
        distances = dest_proj.geometry.distance(origin.geometry)
        
        if not distances.empty:
            # Nearest destination
            min_idx = distances.idxmin()
            min_dist = distances.min()
            result.loc[idx, 'nearest_dist'] = min_dist
            result.loc[idx, 'nearest_id'] = min_idx
            
            # Count destinations within range
            if max_distance is not None:
                within_range = distances[distances <= max_distance]
                result.loc[idx, 'destinations_within_range'] = len(within_range)
                
                # Average distance to destinations within range
                if len(within_range) > 0:
                    result.loc[idx, 'avg_dist_to_destinations'] = within_range.mean()
                else:
                    result.loc[idx, 'avg_dist_to_destinations'] = float('inf')
            else:
                result.loc[idx, 'destinations_within_range'] = len(distances)
                result.loc[idx, 'avg_dist_to_destinations'] = distances.mean()
    
    return result

def _calculate_weighted_distances(origin_proj, dest_proj, result, max_distance):
    """Calculate weighted distances based on destination importance/frequency."""
    # Add columns for weighted distance metrics
    result['nearest_dist'] = float('inf')
    result['nearest_id'] = None
    result['weighted_avg_distance'] = float('inf')
    
    # Check if destinations have weight column
    weight_col = None
    weight_cols = ['trips_per_day', 'frequency', 'weight', 'importance']
    for col in weight_cols:
        if col in dest_proj.columns:
            weight_col = col
            break
    
    for idx, origin in origin_proj.iterrows():
        # Calculate distance to all destinations
        distances = dest_proj.geometry.distance(origin.geometry)
        
        if not distances.empty:
            # Nearest destination (unweighted)
            min_idx = distances.idxmin()
            min_dist = distances.min()
            result.loc[idx, 'nearest_dist'] = min_dist
            result.loc[idx, 'nearest_id'] = min_idx
            
            # Weighted average distance
            if weight_col is not None:
                # Filter by max distance if specified
                if max_distance is not None:
                    valid_mask = distances <= max_distance
                    valid_distances = distances[valid_mask]
                    valid_weights = dest_proj.loc[valid_mask.index, weight_col]
                else:
                    valid_distances = distances
                    valid_weights = dest_proj[weight_col]
                
                # Calculate weighted average (higher weight = more important = effectively closer)
                if len(valid_distances) > 0 and valid_weights.sum() > 0:
                    # Inverse weighting: higher frequency/importance reduces effective distance
                    weighted_distances = valid_distances / (valid_weights + 1)  # +1 to avoid division by zero
                    result.loc[idx, 'weighted_avg_distance'] = weighted_distances.mean()
                else:
                    result.loc[idx, 'weighted_avg_distance'] = float('inf')
            else:
                # No weights available, use simple average
                if max_distance is not None:
                    within_range = distances[distances <= max_distance]
                    if len(within_range) > 0:
                        result.loc[idx, 'weighted_avg_distance'] = within_range.mean()
                    else:
                        result.loc[idx, 'weighted_avg_distance'] = float('inf')
                else:
                    result.loc[idx, 'weighted_avg_distance'] = distances.mean()
    
    return result

def interpolate_points(line_gdf, distance=100):
    """
    Create points along lines at regular intervals.
    
    Useful for things like creating sample points along roads or sidewalks.
    
    Args:
        line_gdf: GeoDataFrame with line geometries
        distance: Distance between points in meters
        
    Returns:
        GeoDataFrame with point geometries
    """
    if line_gdf is None or line_gdf.empty:
        logger.warning("No lines to interpolate points along")
        return gpd.GeoDataFrame(geometry=[], crs=line_gdf.crs if line_gdf is not None else DEFAULT_CRS)
    
    # Convert to projected CRS for accurate interpolation
    analysis_crs = DEFAULT_ANALYSIS_CRS
    line_proj = ensure_crs(line_gdf.copy(), analysis_crs)
    
    try:
        logger.info(f"Creating points every {distance}m along lines")
        
        # Store points and their attributes
        points = []
        attributes = []
        
        # Process each line
        for idx, line in line_proj.iterrows():
            # Get the line's attributes (everything except geometry)
            line_attrs = line.drop('geometry').to_dict()
            line_geom = line.geometry
            
            # Skip invalid geometries
            if not line_geom.is_valid or not isinstance(line_geom, LineString):
                continue
            
            # Calculate how many points we need
            line_length = line_geom.length
            num_points = int(line_length / distance) + 1
            
            # Create points along the line
            for i in range(num_points):
                # Calculate distance along line (don't go past the end)
                dist = min(i * distance, line_length)
                
                # Create point at this distance
                point = line_geom.interpolate(dist)
                
                # Store point and attributes
                points.append(point)
                attributes.append(line_attrs)
        
        # Create GeoDataFrame from the points
        point_gdf = gpd.GeoDataFrame(attributes, geometry=points, crs=analysis_crs)
        
        # Convert back to original coordinate system
        point_gdf = ensure_crs(point_gdf, line_gdf.crs)
        
        logger.info(f"Interpolation complete - {len(point_gdf)} points created")
        return point_gdf
        
    except Exception as e:
        logger.error(f"Point interpolation failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=line_gdf.crs)

def aggregate_to_grid(point_gdf, grid_gdf, value_column, agg_method='count'):
    """
    Aggregate point values to a grid.
    
    Useful for things like counting how many amenities are in each grid cell.
    
    Args:
        point_gdf: GeoDataFrame with point geometries
        grid_gdf: GeoDataFrame with grid cell polygons
        value_column: Column to aggregate
        agg_method: How to aggregate ('count', 'sum', 'mean', 'median', 'min', 'max')
        
    Returns:
        GeoDataFrame with grid cells and aggregated values
    """
    # Check for empty inputs
    if point_gdf is None or point_gdf.empty:
        logger.warning("No points to aggregate")
        return grid_gdf
    
    if grid_gdf is None or grid_gdf.empty:
        logger.warning("No grid to aggregate to")
        return grid_gdf
    
    # Make sure both datasets use the same coordinate system
    point_gdf = ensure_crs(point_gdf, grid_gdf.crs)
    
    try:
        logger.info(f"Aggregating {value_column} to grid using {agg_method}")
        
        # Join points to grid cells
        joined = gpd.sjoin(point_gdf, grid_gdf, how='left', predicate='within')
        
        # Group by grid cell and aggregate
        if agg_method == 'count':
            aggregated = joined.groupby('index_right').size().reset_index(name=f'{value_column}_count')
        else:
            aggregated = joined.groupby('index_right')[value_column].agg(agg_method).reset_index(name=f'{value_column}_{agg_method}')
        
        # Merge aggregated values back to the grid
        result = grid_gdf.merge(aggregated, left_index=True, right_on='index_right', how='left')
        
        # Fill missing values with 0
        if agg_method == 'count':
            result[f'{value_column}_count'] = result[f'{value_column}_count'].fillna(0)
        else:
            result[f'{value_column}_{agg_method}'] = result[f'{value_column}_{agg_method}'].fillna(0)
        
        # Clean up temporary columns
        if 'index_right' in result.columns:
            result = result.drop(columns=['index_right'])
        
        logger.info("Grid aggregation complete")
        return result
        
    except Exception as e:
        logger.error(f"Grid aggregation failed: {e}")
        return grid_gdf