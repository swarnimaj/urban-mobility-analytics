"""
Test script to verify that all required dependencies can be imported.
"""
import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

if __name__ == "__main__":
    modules_to_test = [
        # Core data processing
        "numpy",
        "pandas",
        "geopandas",
        "shapely",
        "pyproj",
        
        # Geospatial analysis
        "osmnx",
        "networkx",
        "folium",
        "pydeck",
        "plotly",
        
        # Data acquisition
        "requests",
        "bs4",  # beautifulsoup4
        "census",
        
        # Visualization
        "streamlit",
        "matplotlib",
        "seaborn",
        
        # Utilities
        "tqdm",
        "dotenv",
    ]
    
    all_passed = all(test_import(module) for module in modules_to_test)
    
    if all_passed:
        print("\n🎉 All dependencies imported successfully!")
        sys.exit(0)
    else:
        print("\n❗ Some dependencies failed to import. Please check your installation.")
        sys.exit(1)