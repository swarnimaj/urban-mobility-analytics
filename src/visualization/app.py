"""
Main Streamlit application for Urban Mobility Analytics.
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import sys
import glob
from pathlib import Path
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import DEFAULT_CITY, DEFAULT_STATE
from src.visualization.transit_visualizations import TransitVisualizationSuite
import matplotlib.pyplot as plt
import matplotlib

# Set page configuration
st.set_page_config(
    page_title="Urban Mobility Analytics",
    page_icon="🚶‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_latest_data(category, file_pattern):
    """
    Load the latest data file for a given category.
    
    Args:
        category: Data category (transit, census, osm, integrated)
        file_pattern: File pattern to match (e.g., "transit_stops_*.geoparquet")
    
    Returns:
        GeoDataFrame or None if no data found
    """
    try:
        data_dir = Path(f"data/processed/{category}")
        if not data_dir.exists():
            st.error(f"Data directory not found: {data_dir}")
            return None
        
        # Find all matching files
        pattern = data_dir / file_pattern
        files = list(Path(".").glob(str(pattern)))
        
        if not files:
            st.warning(f"No {category} data files found matching pattern: {file_pattern}")
            return None
        
        # Get the most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        # Load the data
        data = gpd.read_parquet(latest_file)
        return data
        
    except Exception as e:
        st.error(f"Error loading {category} data: {e}")
        return None

def load_mobility_data():
    """Load the latest mobility index data."""
    return load_latest_data("integrated", "mobility_index_*.geoparquet")

def load_transit_data():
    """Load the latest transit data."""
    return load_latest_data("transit", "transit_stops_*.geoparquet")

def load_census_data():
    """Load the latest census data."""
    return load_latest_data("census", "census_tracts_*.geoparquet")

def load_osm_data():
    """Load the latest OSM data."""
    sidewalks = load_latest_data("osm", "sidewalks_*.geoparquet")
    amenities = load_latest_data("osm", "amenities_*.geoparquet")
    return sidewalks, amenities

# Main title
st.title("Urban Mobility Analytics Platform")
st.markdown("""
This platform visualizes urban mobility equity across neighborhoods, focusing on accessibility
for people with disabilities, aging populations, and car-free commuters.
""")

# Sidebar for controls
st.sidebar.header("Controls")
selected_city = st.sidebar.selectbox("Select City", [DEFAULT_CITY])
selected_state = st.sidebar.selectbox("Select State", [DEFAULT_STATE])

# Load data
with st.spinner("Loading data..."):
    transit_data = load_transit_data()
    mobility_data = load_mobility_data()
    census_data = load_census_data()
    sidewalks_data, amenities_data = load_osm_data()

# Main content
st.header(f"Mobility Analysis for {selected_city}, {selected_state}")

# Display overall metrics if mobility data is available
if mobility_data is not None:
    st.subheader("Overall Mobility Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_mobility = mobility_data['mobility_access_index'].mean()
        st.metric("Average Mobility Index", f"{avg_mobility:.3f}")
    
    with col2:
        total_tracts = len(mobility_data)
        st.metric("Total Census Tracts", total_tracts)
    
    with col3:
        total_population = mobility_data['total_population'].sum()
        st.metric("Total Population", f"{total_population:,.0f}")
    
    with col4:
        high_access = len(mobility_data[mobility_data['mobility_access_index'] >= 0.7])
        st.metric("High Access Tracts", f"{high_access} ({high_access/total_tracts*100:.1f}%)")

# Display tabs for different analyses
tab_transit, tab_sidewalks, tab_amenities, tab_mobility = st.tabs([
    "Transit Access", 
    "Sidewalk Quality", 
    "Amenity Proximity",
    "Mobility Index"
])

with tab_transit:
    st.header("Transit Accessibility Analysis")
    
    if transit_data is not None and census_data is not None:
        # Initialize transit visualization suite
        viz_suite = TransitVisualizationSuite()
        
        # Display transit statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transit Stops", len(transit_data))
        
        with col2:
            if 'wheelchair_accessible' in transit_data.columns:
                accessible_stops = transit_data[transit_data['wheelchair_accessible'] == 'yes'].shape[0]
                pct_accessible = (accessible_stops / len(transit_data)) * 100
                st.metric("Wheelchair Accessible", f"{accessible_stops} ({pct_accessible:.1f}%)")
            else:
                st.metric("Wheelchair Accessible", "Data not available")
        
        with col3:
            if 'trips_per_day' in transit_data.columns:
                avg_trips = transit_data['trips_per_day'].mean()
                st.metric("Avg Trips/Day", f"{avg_trips:.0f}")
            else:
                st.metric("Avg Trips/Day", "Data not available")
        
        with col4:
            if 'avg_headway_minutes' in transit_data.columns:
                avg_headway = transit_data['avg_headway_minutes'].mean()
                st.metric("Avg Headway", f"{avg_headway:.1f} min")
            else:
                st.metric("Avg Headway", "Data not available")
        
        # Create sub-tabs for different transit visualizations
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "🗺️ Interactive Map", 
            "📊 Score Analysis", 
            "📈 Distribution", 
            "🔄 Comparative"
        ])
        
        with subtab1:
            st.subheader("Transit Access Map")
            
            # Check if we have transit scores from mobility data
            display_data = census_data
            score_column = 'transit_access_score'
            
            if mobility_data is not None and 'transit_access_score' in mobility_data.columns:
                display_data = mobility_data
            elif 'transit_access_score' in census_data.columns:
                st.info("📍 Using transit access scores from census data")
            else:
                st.warning("⚠️ No transit access scores found. Showing basic transit stop locations.")
                score_column = None
            
            # Create the comprehensive transit access map
            try:
                matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
                fig = viz_suite.create_transit_access_map(
                    neighborhoods=display_data,
                    transit_stops=transit_data,
                    score_column=score_column
                )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error creating transit access map: {e}")
                
                # Fallback to the original PyDeck map
                st.subheader("Transit Stops (Fallback View)")
                
                # Prepare data for visualization
                if 'wheelchair_accessible' in transit_data.columns:
                    def get_color(accessibility):
                        if accessibility == 'yes':
                            return [0, 255, 0, 200]  # Green for accessible
                        elif accessibility == 'no':
                            return [255, 0, 0, 200]  # Red for not accessible
                        else:
                            return [128, 128, 128, 200]  # Gray for unknown
                    
                    transit_data['color'] = transit_data['wheelchair_accessible'].apply(get_color)
                else:
                    transit_data['color'] = [0, 100, 255, 200]  # Blue
                
                # Create the layer
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    transit_data,
                    get_position=['stop_lon', 'stop_lat'],
                    get_color='color',
                    get_radius=50,
                    pickable=True
                )
                
                # Set the initial view (Seattle area)
                view_state = pdk.ViewState(
                    latitude=47.6062,
                    longitude=-122.3321,
                    zoom=10,
                    pitch=0
                )
                
                # Create tooltip
                tooltip = {
                    "html": """
                    <b>{stop_name}</b><br/>
                    <b>ID:</b> {stop_id}<br/>
                    <b>Wheelchair:</b> {wheelchair_accessible}<br/>
                    <b>Agency:</b> {agency_name}<br/>
                    <b>Trips/Day:</b> {trips_per_day}<br/>
                    <b>Headway:</b> {avg_headway_minutes:.1f} min
                    """,
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                }
                
                # Render the map
                st.pydeck_chart(pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip
                ))
        
        with subtab2:
            st.subheader("Transit Score Analysis")
            
            # Check for new scoring system data
            if mobility_data is not None and 'transit_access_score' in mobility_data.columns:
                score_data = mobility_data
                
                # First show the overall transit access score statistics
                st.write("**Overall Transit Access Scores:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_score = score_data['transit_access_score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}")
                
                with col2:
                    max_score = score_data['transit_access_score'].max()
                    st.metric("Best Score", f"{max_score:.1f}")
                
                with col3:
                    min_score = score_data['transit_access_score'].min()
                    st.metric("Lowest Score", f"{min_score:.1f}")
                
                with col4:
                    std_score = score_data['transit_access_score'].std()
                    st.metric("Score Variation", f"{std_score:.1f}")
                
                # Display detailed score breakdown if available
                if 'distance_score' in score_data.columns:
                    st.write("**Score Components Breakdown:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_distance = score_data['distance_score'].mean()
                        st.metric("Distance Score", f"{avg_distance:.1f}")
                    
                    with col2:
                        if 'frequency_score' in score_data.columns:
                            avg_frequency = score_data['frequency_score'].mean()
                            st.metric("Frequency Score", f"{avg_frequency:.1f}")
                        else:
                            st.metric("Frequency Score", "N/A")
                    
                    with col3:
                        if 'accessibility_score' in score_data.columns:
                            avg_accessibility = score_data['accessibility_score'].mean()
                            st.metric("Accessibility Score", f"{avg_accessibility:.1f}")
                        else:
                            st.metric("Accessibility Score", "N/A")
                    
                    with col4:
                        if 'coverage_score' in score_data.columns:
                            avg_coverage = score_data['coverage_score'].mean()
                            st.metric("Coverage Score", f"{avg_coverage:.1f}")
                        else:
                            st.metric("Coverage Score", "N/A")
                else:
                    st.info("💡 Component score breakdown not available. Run the latest data pipeline for detailed breakdowns.")
                
                # Show transit access score distribution
                st.write("**Score Distribution:**")
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    scores = score_data['transit_access_score'].dropna()
                    ax.hist(scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                    ax.set_title('Distribution of Transit Access Scores')
                    ax.set_xlabel('Transit Access Score')
                    ax.set_ylabel('Number of Areas')
                    ax.grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_score = scores.mean()
                    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                             label=f'Average: {mean_score:.1f}')
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error creating score distribution: {e}")
                
                # Distance vs Frequency scatter plot
                try:
                    if 'nearest_stop_distance' in score_data.columns and 'frequency_score' in score_data.columns:
                        st.write("**Distance vs Frequency Analysis:**")
                        fig = viz_suite.create_distance_frequency_scatter(score_data)
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {e}")
                
                # Show top and bottom performing areas
                st.write("**Performance Analysis:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 5 Best Transit Access:**")
                    if 'geoid' in score_data.columns:
                        top_areas = score_data.nlargest(5, 'transit_access_score')[['geoid', 'transit_access_score']]
                        top_areas.columns = ['Area ID', 'Score']
                        st.dataframe(top_areas, hide_index=True)
                    else:
                        top_scores = score_data['transit_access_score'].nlargest(5)
                        st.write(f"Top scores: {', '.join([f'{score:.1f}' for score in top_scores])}")
                
                with col2:
                    st.write("**Bottom 5 Transit Access:**")
                    if 'geoid' in score_data.columns:
                        bottom_areas = score_data.nsmallest(5, 'transit_access_score')[['geoid', 'transit_access_score']]
                        bottom_areas.columns = ['Area ID', 'Score']
                        st.dataframe(bottom_areas, hide_index=True)
                    else:
                        bottom_scores = score_data['transit_access_score'].nsmallest(5)
                        st.write(f"Lowest scores: {', '.join([f'{score:.1f}' for score in bottom_scores])}")
                    
            else:
                st.info("💡 Run the new transit scoring system to see detailed score analysis here!")
                
                # Show basic statistics even when comprehensive scores aren't available
                st.write("**Current Transit Data Analysis:**")
                
                # Basic transit metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'trips_per_day' in transit_data.columns:
                        avg_trips = transit_data['trips_per_day'].mean()
                        max_trips = transit_data['trips_per_day'].max()
                        st.metric("Average Service Frequency", f"{avg_trips:.0f} trips/day", f"Max: {max_trips:.0f}")
                    
                    if 'avg_headway_minutes' in transit_data.columns:
                        avg_headway = transit_data['avg_headway_minutes'].mean()
                        min_headway = transit_data['avg_headway_minutes'].min()
                        st.metric("Average Headway", f"{avg_headway:.1f} min", f"Best: {min_headway:.1f} min")
                
                with col2:
                    # Accessibility breakdown
                    if 'wheelchair_accessible' in transit_data.columns:
                        accessible_count = (transit_data['wheelchair_accessible'] == 'yes').sum()
                        total_count = len(transit_data)
                        pct_accessible = (accessible_count / total_count) * 100
                        st.metric("Accessibility Coverage", f"{pct_accessible:.1f}%", f"{accessible_count}/{total_count} stops")
                    
                    # Agency diversity
                    if 'agency_name' in transit_data.columns:
                        agency_count = transit_data['agency_name'].nunique()
                        st.metric("Transit Agencies", f"{agency_count} agencies")
                
                # Agency breakdown table
                if 'agency_name' in transit_data.columns:
                    st.write("**Service by Agency:**")
                    agency_stats = transit_data.groupby('agency_name').agg({
                        'stop_id': 'count',
                        'wheelchair_accessible': lambda x: (x == 'yes').sum(),
                        'trips_per_day': 'mean',
                        'avg_headway_minutes': 'mean'
                    }).round(2)
                    agency_stats.columns = ['Total Stops', 'Accessible Stops', 'Avg Trips/Day', 'Avg Headway (min)']
                    st.dataframe(agency_stats)
                
                # Basic service quality chart
                if 'trips_per_day' in transit_data.columns:
                    st.write("**Service Frequency Distribution:**")
                    st.bar_chart(transit_data['trips_per_day'].value_counts().head(10))
        
        with subtab3:
            st.subheader("Score Distribution Analysis")
            
            # Create distribution plots
            if mobility_data is not None and 'transit_access_score' in mobility_data.columns:
                try:
                    # Use a simpler approach to avoid KDE issues
                    st.write("**Transit Access Score Distribution:**")
                    
                    # Create a simple histogram
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    scores = mobility_data['transit_access_score'].dropna()
                    if len(scores) > 0:
                        # Simple histogram
                        ax.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_title('Distribution of Transit Access Scores')
                        ax.set_xlabel('Transit Access Score')
                        ax.set_ylabel('Frequency')
                        ax.grid(True, alpha=0.3)
                        
                        # Add mean and median lines
                        mean_score = scores.mean()
                        median_score = scores.median()
                        ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.1f}')
                        ax.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.1f}')
                        ax.legend()
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("No valid transit access scores found")
                    
                    # Show statistics
                    st.write("**Transit Access Score Statistics:**")
                    score_stats = mobility_data['transit_access_score'].describe()
                    st.dataframe(score_stats)
                    
                except Exception as e:
                    st.error(f"Error creating distribution plots: {e}")
                    st.write("Showing basic statistics instead:")
                    if 'transit_access_score' in mobility_data.columns:
                        score_stats = mobility_data['transit_access_score'].describe()
                        st.dataframe(score_stats)
                        
            else:
                st.info("💡 Transit score distribution will be available after running the new scoring system!")
                
                # Show basic transit data distribution instead
                st.write("**Current Transit Data Analysis:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'trips_per_day' in transit_data.columns:
                        st.write("**Service Frequency Distribution:**")
                        trips_data = transit_data['trips_per_day'].dropna()
                        if len(trips_data) > 0:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(trips_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                            ax.set_title('Distribution of Trips per Day')
                            ax.set_xlabel('Trips per Day')
                            ax.set_ylabel('Number of Stops')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Statistics
                            trips_stats = trips_data.describe()
                            st.dataframe(trips_stats)
                
                with col2:
                    if 'avg_headway_minutes' in transit_data.columns:
                        st.write("**Headway Distribution:**")
                        headway_data = transit_data['avg_headway_minutes'].dropna()
                        if len(headway_data) > 0:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.hist(headway_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                            ax.set_title('Distribution of Average Headway')
                            ax.set_xlabel('Headway (minutes)')
                            ax.set_ylabel('Number of Stops')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Statistics
                            headway_stats = headway_data.describe()
                            st.dataframe(headway_stats)
        
        with subtab4:
            st.subheader("Comparative Analysis")
            
            if mobility_data is not None and 'transit_access_score' in mobility_data.columns:
                try:
                    # Create accessibility categories for comparison
                    if 'accessibility_category' not in mobility_data.columns:
                        mobility_data['accessibility_category'] = pd.cut(
                            mobility_data['transit_access_score'],
                            bins=[0, 25, 50, 75, 100],
                            labels=['Poor', 'Fair', 'Good', 'Excellent']
                        )
                    
                    fig = viz_suite.create_comparative_analysis(
                        mobility_data,
                        grouping_col='accessibility_category',
                        score_col='transit_access_score'
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show category breakdown
                    st.write("**Access Categories:**")
                    category_counts = mobility_data['accessibility_category'].value_counts()
                    category_pcts = (category_counts / len(mobility_data) * 100).round(1)
                    
                    for category in ['Poor', 'Fair', 'Good', 'Excellent']:
                        if category in category_counts.index:
                            count = category_counts[category]
                            pct = category_pcts[category]
                            st.metric(f"{category} Access", f"{count} tracts ({pct}%)")
                    
                except Exception as e:
                    st.error(f"Error creating comparative analysis: {e}")
            else:
                st.info("💡 Comprehensive comparative analysis will be available after running the new scoring system!")
                
                # Show basic comparisons with current data
                st.write("**Current Transit Data Comparisons:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accessibility comparison
                    if 'wheelchair_accessible' in transit_data.columns:
                        st.write("**Wheelchair Accessibility:**")
                        access_counts = transit_data['wheelchair_accessible'].value_counts()
                        
                        # Create a more informative chart
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['green' if x == 'yes' else 'red' if x == 'no' else 'gray' for x in access_counts.index]
                        bars = ax.bar(access_counts.index, access_counts.values, color=colors, alpha=0.7)
                        ax.set_title('Transit Stop Accessibility')
                        ax.set_ylabel('Number of Stops')
                        ax.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, access_counts.values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                                   str(value), ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Show percentages
                        total = access_counts.sum()
                        for access_type, count in access_counts.items():
                            pct = (count / total) * 100
                            st.metric(f"{access_type.title()} Stops", f"{count} ({pct:.1f}%)")
                
                with col2:
                    # Agency comparison
                    if 'agency_name' in transit_data.columns:
                        st.write("**Service by Agency:**")
                        agency_counts = transit_data['agency_name'].value_counts().head(5)
                        
                        fig, ax = plt.subplots(figsize=(8, 5))
                        bars = ax.bar(range(len(agency_counts)), agency_counts.values, 
                                     color=plt.cm.Set3(range(len(agency_counts))))
                        ax.set_title('Top 5 Agencies by Stop Count')
                        ax.set_ylabel('Number of Stops')
                        ax.set_xticks(range(len(agency_counts)))
                        ax.set_xticklabels(agency_counts.index, rotation=45, ha='right')
                        ax.grid(True, alpha=0.3)
                        
                        # Add value labels
                        for bar, value in zip(bars, agency_counts.values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                                   str(value), ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                
                # Service frequency categories
                if 'trips_per_day' in transit_data.columns:
                    st.write("**Service Frequency Categories:**")
                    
                    # Create frequency categories
                    frequency_data = transit_data['trips_per_day'].dropna()
                    if len(frequency_data) > 0:
                        freq_categories = pd.cut(frequency_data, 
                                               bins=[0, 20, 50, 100, float('inf')], 
                                               labels=['Low (≤20)', 'Medium (21-50)', 'High (51-100)', 'Very High (>100)'])
                        freq_counts = freq_categories.value_counts()
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['red', 'orange', 'lightgreen', 'darkgreen']
                        bars = ax.bar(freq_counts.index, freq_counts.values, color=colors[:len(freq_counts)])
                        ax.set_title('Transit Stops by Service Frequency Category')
                        ax.set_ylabel('Number of Stops')
                        ax.set_xlabel('Trips per Day Category')
                        plt.xticks(rotation=45)
                        ax.grid(True, alpha=0.3)
                        
                        # Add percentages
                        total = freq_counts.sum()
                        for bar, value in zip(bars, freq_counts.values):
                            pct = (value / total) * 100
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                                   f'{value}\n({pct:.1f}%)', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
        
    else:
        if transit_data is None:
            st.error("🚫 Transit data not available. Please run the data pipeline first.")
        if census_data is None:
            st.error("🚫 Census data not available. Please run the data pipeline first.")
        
        st.info("""
        **To see the enhanced transit accessibility analysis:**
        1. Run the data acquisition pipeline to fetch transit and census data
        2. Run the data integration process to calculate comprehensive transit scores
        3. The new Sprint 7 transit scoring system will provide detailed accessibility metrics
        """)

with tab_sidewalks:
    st.header("Sidewalk Quality")
    
    if sidewalks_data is not None:
        st.metric("Total Sidewalk Segments", len(sidewalks_data))
        
        if 'length_km' in sidewalks_data.columns:
            total_length = sidewalks_data['length_km'].sum()
            st.metric("Total Sidewalk Length", f"{total_length:.1f} km")
        
        # Add sidewalk visualization here
        st.info("Sidewalk quality visualization will be implemented here.")
    else:
        st.error("Sidewalk data not available. Please run the data pipeline first.")

with tab_amenities:
    st.header("Amenity Proximity")
    
    if amenities_data is not None:
        st.metric("Total Amenities", len(amenities_data))
        
        if 'amenity' in amenities_data.columns:
            amenity_types = amenities_data['amenity'].value_counts()
            st.write("**Amenity Types:**")
            st.dataframe(amenity_types.head(10))
        
        # Add amenity visualization here
        st.info("Amenity proximity visualization will be implemented here.")
    else:
        st.error("Amenity data not available. Please run the data pipeline first.")

with tab_mobility:
    st.header("Mobility Accessibility Index")
    
    if mobility_data is not None:
        # Display mobility index map
        st.subheader("Mobility Index by Census Tract")
        
        # Create choropleth map of mobility index
        layer = pdk.Layer(
            "GeoJsonLayer",
            mobility_data,
            get_fill_color="[255 * mobility_access_index, 255 * (1 - mobility_access_index), 0, 200]",
            get_line_color=[255, 255, 255],
            get_line_width=1,
            pickable=True
        )
        
        # Set the initial view
        view_state = pdk.ViewState(
            latitude=47.6062,
            longitude=-122.3321,
            zoom=9,
            pitch=0
        )
        
        # Create tooltip
        tooltip = {
            "html": """
            <b>Census Tract: {geoid}</b><br/>
            <b>Mobility Index:</b> {mobility_access_index:.3f}<br/>
            <b>Population:</b> {total_population}<br/>
            <b>Transit Stops:</b> {stop_count}<br/>
            <b>Transit Score:</b> {transit_access_score:.1f}<br/>
            <b>Sidewalk Score:</b> {sidewalk_quality_score:.1f}<br/>
            <b>Amenity Score:</b> {amenity_proximity_score:.1f}
            """,
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
        
        # Render the map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))
        
        # Display mobility statistics
        st.subheader("Mobility Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Mobility Index Distribution:**")
            mobility_stats = mobility_data['mobility_access_index'].describe()
            st.dataframe(mobility_stats)
        
        with col2:
            st.write("**Score Components:**")
            score_components = {
                'Transit Access': mobility_data['transit_access_score'].mean(),
                'Sidewalk Quality': mobility_data['sidewalk_quality_score'].mean(),
                'Amenity Proximity': mobility_data['amenity_proximity_score'].mean()
            }
            for component, score in score_components.items():
                st.metric(component, f"{score:.1f}")
        
        with col3:
            st.write("**Accessibility Categories:**")
            low_access = len(mobility_data[mobility_data['mobility_access_index'] <= 0.3])
            medium_access = len(mobility_data[(mobility_data['mobility_access_index'] > 0.3) & (mobility_data['mobility_access_index'] < 0.7)])
            high_access = len(mobility_data[mobility_data['mobility_access_index'] >= 0.7])
            
            st.metric("Low Access (≤0.3)", f"{low_access} ({low_access/len(mobility_data)*100:.1f}%)")
            st.metric("Medium Access (0.3-0.7)", f"{medium_access} ({medium_access/len(mobility_data)*100:.1f}%)")
            st.metric("High Access (≥0.7)", f"{high_access} ({high_access/len(mobility_data)*100:.1f}%)")
    
    else:
        st.error("Mobility data not available. Please run the data pipeline first.")

# Footer
st.markdown("---")
st.markdown("Urban Mobility Analytics Platform | Created for accessibility and equity analysis")