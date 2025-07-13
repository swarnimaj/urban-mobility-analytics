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
tab1, tab2, tab3, tab4 = st.tabs([
    "Transit Access", 
    "Sidewalk Quality", 
    "Amenity Proximity",
    "Street Connectivity"
])

with tab1:
    st.markdown("### Transit Access Score")
    st.info("Transit access analysis will be displayed here.")
    
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