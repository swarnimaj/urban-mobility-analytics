"""
Main Streamlit application for Urban Mobility Analytics.
"""
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import sys
from pathlib import Path

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

# Main content
st.header(f"Mobility Analysis for {selected_city}, {selected_state}")

# Placeholder for map
st.subheader("Mobility Accessibility Map")
st.info("Map will be displayed here once data is loaded.")

# Display tabs for different analyses
tab_transit, tab2, tab3, tab4 = st.tabs([
    "Transit Access", 
    "Sidewalk Quality", 
    "Amenity Proximity",
    "Street Connectivity"
])

with tab_transit:
    st.header("Transit Accessibility")
    
    # Load transit data if it exists
    transit_data_path = Path("data/processed/gtfs/king_county_metro/stops.geojson")
    if transit_data_path.exists():
        # Load the data
        transit_stops = gpd.read_file(transit_data_path)
        
        # Display basic statistics
        st.metric("Total Transit Stops", len(transit_stops))
        
        if 'wheelchair_accessible' in transit_stops.columns:
            accessible_stops = transit_stops[transit_stops['wheelchair_accessible'] == 'yes'].shape[0]
            pct_accessible = (accessible_stops / len(transit_stops)) * 100
            st.metric("Wheelchair Accessible Stops", f"{accessible_stops} ({pct_accessible:.1f}%)")
        
        # Display map
        st.subheader("Transit Stops Map")
        
        # Use Pydeck for an interactive map
        import pydeck as pdk
        
        # Define color based on wheelchair accessibility
        transit_stops['color'] = transit_stops['wheelchair_accessible'].map({
            'yes': [0, 255, 0, 200],  # Green
            'no': [255, 0, 0, 200],   # Red
            'unknown': [128, 128, 128, 200]  # Gray
        }).fillna([128, 128, 128, 200])
        
        # Create the layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            transit_stops,
            get_position=['stop_lon', 'stop_lat'],
            get_color='color',
            get_radius=100,
            pickable=True
        )
        
        # Set the initial view
        view_state = pdk.ViewState(
            latitude=transit_stops['stop_lat'].mean(),
            longitude=transit_stops['stop_lon'].mean(),
            zoom=10,
            pitch=0
        )
        
        # Render the map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{stop_name}\nWheelchair Accessible: {wheelchair_accessible}"}
        ))
    else:
        st.info("Transit data not yet processed. Run the GTFS processor to see transit accessibility data.")
    
with tab2:
    st.markdown("### Sidewalk Quality Score")
    st.info("Sidewalk quality analysis will be displayed here.")
    
with tab3:
    st.markdown("### Amenity Proximity Score")
    st.info("Amenity proximity analysis will be displayed here.")
    
with tab4:
    st.markdown("### Street Connectivity Score")
    st.info("Street connectivity analysis will be displayed here.")

# Footer
st.markdown("---")
st.markdown("Urban Mobility Analytics Platform | Created for accessibility and equity analysis")