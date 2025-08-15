"""
Example script to fetch and process OpenStreetMap data for a target city.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
from pathlib import Path
import logging
from osm_downloader import OSMDownloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "osm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_process_osm_data(place_name="Seattle, Washington", run_full_analysis=False, max_amenities_for_full_analysis=500):
    """
    Main function to fetch and process OSM data for urban mobility analysis.
    
    This function downloads street networks, sidewalks, and amenities, then performs
    accessibility analysis. For large datasets, it skips computationally expensive
    analyses to maintain performance.
    """
    logger.info(f"Starting OSM data processing for {place_name}")
    
    try:
        # Initialize downloader and get core data
        downloader = OSMDownloader(place_name=place_name)
        
        # Download street network (all types: drive, walk, bike)
        logger.info("Downloading street network...")
        street_network = downloader.get_street_network(network_type='all')
        
        # Download pedestrian infrastructure (sidewalks, crossings, curb ramps)
        logger.info("Downloading sidewalk and crossing data...")
        sidewalk_data = downloader.get_sidewalk_data()
        
        # Handle download failures gracefully
        if sidewalk_data is None:
            logger.warning("Sidewalk download failed - creating empty dataset")
            sidewalk_data = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Download points of interest (schools, hospitals, etc.)
        logger.info("Downloading amenities...")
        amenities = downloader.get_amenities()
        
        if amenities is None:
            logger.warning("Amenities download failed - creating empty dataset")
            amenities = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Filter amenities for accessibility features (wheelchair access, etc.)
        logger.info("Analyzing amenity accessibility...")
        accessible_amenities = downloader.filter_accessible_amenities(amenities)
        
        # Calculate sidewalk coverage statistics
        logger.info("Calculating sidewalk coverage...")
        sidewalk_coverage = downloader.calculate_sidewalk_coverage(
            street_network=street_network,
            sidewalk_data=sidewalk_data
        )
        
        # Decide whether to run full analysis based on dataset size
        should_run_full = run_full_analysis or len(amenities) <= max_amenities_for_full_analysis
        
        # Full accessibility analysis (computationally expensive)
        if should_run_full:
            logger.info("Running full accessibility analysis...")
            amenities_with_access = downloader.analyze_amenity_accessibility(
                amenities=amenities,
                sidewalk_data=sidewalk_data,
                street_network=street_network
            )
        else:
            logger.info(f"Skipping full analysis for {len(amenities)} amenities (performance optimization)")
            amenities_with_access = accessible_amenities.copy()
            amenities_with_access['walkability_score'] = 0
            amenities_with_access['walkability_category'] = 'unknown'
        
        # Identify mobility barriers (missing sidewalks, etc.)
        if should_run_full:
            logger.info("Identifying mobility barriers...")
            mobility_barriers = downloader.identify_mobility_barriers(
                street_network=street_network,
                sidewalk_data=sidewalk_data
            )
        else:
            logger.info("Skipping mobility barrier analysis (performance optimization)")
            mobility_barriers = None
        
        # Assess data quality and completeness
        if should_run_full:
            logger.info("Assessing data quality...")
            data_quality = downloader.assess_osm_data_quality()
        else:
            logger.info("Skipping detailed quality assessment (performance optimization)")
            data_quality = {
                'area_km2': 0,
                'node_count': len(street_network.nodes),
                'edge_count': len(street_network.edges),
                'amenity_count': len(amenities),
                'overall_quality_score': 0
            }
        
        # Calculate walking accessibility areas (isochrones)
        if should_run_full and not amenities.empty:
            logger.info("Calculating walking accessibility areas...")
            try:
                # Find a good starting point for isochrones
                schools = amenities[amenities['amenity'] == 'school']
                hospitals = amenities[amenities['amenity'] == 'hospital']
                
                if not schools.empty:
                    origin_point = schools.iloc[0].geometry
                    origin_name = schools.iloc[0].get('name', 'School')
                elif not hospitals.empty:
                    origin_point = hospitals.iloc[0].geometry
                    origin_name = hospitals.iloc[0].get('name', 'Hospital')
                else:
                    origin_point = amenities.iloc[0].geometry
                    origin_name = amenities.iloc[0].get('name', 'Amenity')
                
                logger.info(f"Calculating isochrones from {origin_name}")
                isochrones = downloader.calculate_isochrones(
                    origin_point=origin_point,
                    travel_times=[5, 10, 15]  # 5, 10, 15 minute walks
                )
            except ImportError:
                logger.warning("Pandana not installed - skipping isochrones. Install with: pip install pandana")
                isochrones = None
        else:
            logger.info("Skipping isochrone calculation (performance optimization)")
            isochrones = None
        
        # Save all processed data to files
        place_name_slug = place_name.lower().replace(", ", "_").replace(" ", "_")
        output_dir = OUTPUT_DIR / place_name_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save street network (GraphML format for NetworkX compatibility)
        try:
            street_network_file = downloader.save_processed_data(
                street_network, "street_network", output_dir, "graphml"
            )
            logger.info(f"Saved street network to {street_network_file}")
        except ImportError as e:
            if "lxml" in str(e):
                logger.warning("lxml not installed - skipping GraphML export. Install with: pip install lxml")
            else:
                raise
        
        # Save pedestrian infrastructure data
        sidewalk_file = downloader.save_processed_data(
            sidewalk_data, "sidewalks", output_dir, "geojson"
        )
        logger.info(f"Saved sidewalk data to {sidewalk_file}")
        
        # Save points of interest with accessibility info
        amenities_file = downloader.save_processed_data(
            accessible_amenities, "amenities", output_dir, "geojson"
        )
        logger.info(f"Saved amenities to {amenities_file}")
        
        # Save analysis results
        coverage_file = downloader.save_processed_data(
            sidewalk_coverage, "sidewalk_coverage", output_dir, "json"
        )
        logger.info(f"Saved sidewalk coverage stats to {coverage_file}")
        
        access_file = downloader.save_processed_data(
            amenities_with_access, "amenities_accessibility", output_dir, "geojson"
        )
        logger.info(f"Saved accessibility analysis to {access_file}")
        
        # Save optional analysis results
        if mobility_barriers is not None and not mobility_barriers.empty:
            barriers_file = downloader.save_processed_data(
                mobility_barriers, "mobility_barriers", output_dir, "geojson"
            )
            logger.info(f"Saved mobility barriers to {barriers_file}")
        
        quality_file = downloader.save_processed_data(
            data_quality, "data_quality", output_dir, "json"
        )
        logger.info(f"Saved data quality assessment to {quality_file}")
        
        if isochrones is not None and not isochrones.empty:
            isochrones_file = downloader.save_processed_data(
                isochrones, "isochrones", output_dir, "geojson"
            )
            logger.info(f"Saved walking accessibility areas to {isochrones_file}")
        
        # Create visualizations for smaller datasets
        if should_run_full and len(amenities) <= 1000:
            logger.info("Creating visualizations")
            create_visualizations(
                street_network=street_network,
                sidewalk_data=sidewalk_data,
                amenities=amenities_with_access,
                mobility_barriers=mobility_barriers,
                isochrones=isochrones,
                place_name=place_name,
                output_dir=output_dir
            )
        else:
            logger.info("Skipping visualizations for performance")
        
        return {
            'street_network': street_network,
            'sidewalk_data': sidewalk_data,
            'amenities': amenities,
            'accessible_amenities': accessible_amenities,
            'sidewalk_coverage': sidewalk_coverage,
            'amenities_with_access': amenities_with_access,
            'mobility_barriers': mobility_barriers,
            'data_quality': data_quality,
            'isochrones': isochrones
        }
        
    except Exception as e:
        logger.error(f"Error processing OSM data: {e}")
        raise

# Create visualizations
def create_visualizations(street_network, sidewalk_data, amenities, 
                                 mobility_barriers=None, isochrones=None,
                                 place_name="", output_dir=None):
    """
    Create extended visualizations of the OSM data including new features.
    
    Args:
        street_network: NetworkX graph of the street network
        sidewalk_data: GeoDataFrame of sidewalks and crossings
        amenities: GeoDataFrame of amenities with accessibility analysis
        mobility_barriers: GeoDataFrame of mobility barriers
        isochrones: GeoDataFrame of isochrones
        place_name: Name of the place
        output_dir: Directory to save visualizations
    """
    try:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        
        # Plot street network
        ox.plot_graph(street_network, ax=axes[0, 0], node_size=0, edge_linewidth=0.5)
        axes[0, 0].set_title(f"Street Network in {place_name}")
        
        # Plot sidewalks and crossings
        if not sidewalk_data.empty:
            # Filter for different infrastructure types
            sidewalks = sidewalk_data[sidewalk_data['infrastructure_type'] == 'sidewalk']
            crossings = sidewalk_data[sidewalk_data['infrastructure_type'] == 'crossing']
            curb_ramps = sidewalk_data[sidewalk_data['infrastructure_type'] == 'curb_ramp']
            
            # Plot each type with different colors
            if not sidewalks.empty:
                sidewalks.plot(ax=axes[0, 1], color='blue', linewidth=0.5, label='Sidewalks')
            if not crossings.empty:
                crossings.plot(ax=axes[0, 1], color='red', linewidth=0.5, label='Crossings')
            if not curb_ramps.empty:
                curb_ramps.plot(ax=axes[0, 1], color='green', markersize=3, label='Curb Ramps')
            
            axes[0, 1].set_title(f"Sidewalks and Crossings in {place_name}")
            axes[0, 1].legend()
        
        # Plot amenities by type
        if not amenities.empty and 'amenity' in amenities.columns:
            # Get top 5 amenity types by count
            top_amenities = amenities['amenity'].value_counts().head(5).index
            
            # Plot each type with different colors
            for amenity_type in top_amenities:
                subset = amenities[amenities['amenity'] == amenity_type]
                subset.plot(ax=axes[1, 0], markersize=5, label=amenity_type)
            
            axes[1, 0].set_title(f"Top Amenities in {place_name}")
            axes[1, 0].legend()
        
        # Plot amenities by walkability score
        if not amenities.empty and 'walkability_score' in amenities.columns:
            amenities.plot(
                ax=axes[1, 1],
                column='walkability_score',
                cmap='RdYlGn',
                legend=True,
                markersize=5
            )
            axes[1, 1].set_title(f"Amenity Walkability in {place_name}")
        
        # NEW: Plot mobility barriers
        if mobility_barriers is not None and not mobility_barriers.empty:
            # Plot different barrier types with different colors
            barrier_types = mobility_barriers['barrier_type'].unique()
            for barrier_type in barrier_types:
                subset = mobility_barriers[mobility_barriers['barrier_type'] == barrier_type]
                subset.plot(
                    ax=axes[2, 0],
                    label=barrier_type.replace('_', ' ').title(),
                    markersize=5 if 'Point' in subset.geometry.type.iloc[0] else None,
                    linewidth=2 if 'LineString' in subset.geometry.type.iloc[0] else None
                )
            
            axes[2, 0].set_title(f"Mobility Barriers in {place_name}")
            axes[2, 0].legend()
        else:
            axes[2, 0].set_title(f"No Mobility Barriers Identified in {place_name}")
        
        # NEW: Plot isochrones
        if isochrones is not None and not isochrones.empty:
            # Use a colormap for different times
            colors = plt.cm.viridis(np.linspace(0, 1, len(isochrones)))
            
            # Plot each isochrone with a different color
            for i, (idx, isochrone) in enumerate(isochrones.iterrows()):
                gpd.GeoDataFrame(geometry=[isochrone.geometry], crs="EPSG:4326").plot(
                    ax=axes[2, 1],
                    color=colors[i],
                    alpha=0.5,
                    label=f"{isochrone['time']} min"
                )
            
            axes[2, 1].set_title(f"Walking Isochrones in {place_name}")
            axes[2, 1].legend()
        else:
            axes[2, 1].set_title(f"No Isochrones Available for {place_name}")
        
        # Save the figure
        plt.tight_layout()
        fig.savefig(output_dir / f"{place_name.lower().replace(', ', '_').replace(' ', '_')}_osm_extended_visualization.png", dpi=300)
        logger.info(f"Saved extended visualization to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating extended visualizations: {e}")

if __name__ == "__main__":
    # Example usage options:
    
    # Option 1: Fast analysis (current default)
    print("Running fast analysis for Seattle...")
    result = fetch_and_process_osm_data("Seattle, Washington")
    
    # Option 2: Full analysis for a smaller area (uncomment to use)
    # print("Running full analysis for a smaller area...")
    # result = fetch_and_process_osm_data("Bellevue, Washington", run_full_analysis=True)
    
    # Option 3: Force full analysis regardless of size (uncomment to use)
    # print("Running full analysis for Seattle (will take much longer)...")
    # result = fetch_and_process_osm_data("Seattle, Washington", run_full_analysis=True)
    
    # Print summary statistics
    if result:
        print("\nSummary Statistics:")
        print(f"Street Network: {len(result['street_network'].nodes)} nodes, {len(result['street_network'].edges)} edges")
        print(f"Sidewalks and Crossings: {len(result['sidewalk_data'])} features")
        print(f"Amenities: {len(result['amenities'])} total, {len(result['accessible_amenities'])} with accessibility info")
        
        if 'sidewalk_coverage' in result:
            coverage = result['sidewalk_coverage']
            print(f"\nSidewalk Coverage:")
            print(f"Road Length: {coverage['road_length_km']:.2f} km")
            print(f"Sidewalk Length: {coverage['sidewalk_length_km']:.2f} km")
            print(f"Coverage Ratio: {coverage['sidewalk_coverage_percent']:.2f}%")
            print(f"Crossings per Intersection: {coverage['crossings_per_intersection']:.2f}")
        
        # Show what analyses were run
        print(f"\nAnalyses Completed:")
        print(f"✓ Basic data extraction")
        print(f"✓ Sidewalk coverage calculation")
        print(f"✓ Amenity accessibility: {'✓' if result['amenities_with_access']['walkability_score'].sum() > 0 else '✗ (skipped)'}")
        print(f"✓ Mobility barriers: {'✓' if result['mobility_barriers'] is not None else '✗ (skipped)'}")
        print(f"✓ Data quality assessment: {'✓' if result['data_quality']['overall_quality_score'] > 0 else '✗ (simplified)'}")
        print(f"✓ Isochrones: {'✓' if result['isochrones'] is not None else '✗ (skipped)'}")
        
        print(f"\nTo run full analysis on smaller areas, use:")
        print(f"fetch_and_process_osm_data('Smaller City, State', run_full_analysis=True)")
        print(f"Or install pandana for isochrone analysis: pip install pandana")