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
    st.header("Transit Accessibility")
    
    if transit_data is not None:
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
        
        # Display map
        st.subheader("Transit Stops Map")
        
        # Prepare data for visualization
        if 'wheelchair_accessible' in transit_data.columns:
            # Define color based on wheelchair accessibility
            def get_color(accessibility):
                if accessibility == 'yes':
                    return [0, 255, 0, 200]  # Green for accessible
                elif accessibility == 'no':
                    return [255, 0, 0, 200]  # Red for not accessible
                else:
                    return [128, 128, 128, 200]  # Gray for unknown
            
            transit_data['color'] = transit_data['wheelchair_accessible'].apply(get_color)
        else:
            # Default color if accessibility data not available
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
        
        # Display detailed statistics
        st.subheader("Transit Service Analysis")
        
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
        
    else:
        st.error("Transit data not available. Please run the data pipeline first.")

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