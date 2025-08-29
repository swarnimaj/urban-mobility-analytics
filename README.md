# Urban Mobility Analytics Platform

A comprehensive geospatial analytics platform for visualizing urban mobility equity across neighborhoods, focusing on accessibility for people with disabilities, aging populations, and car-free commuters.

## 🎯 Project Overview

This platform analyzes and visualizes how equitably urban infrastructure supports mobility across different neighborhoods by:
- **Calculating multi-dimensional Mobility Accessibility Index (MAI)** with normalized 0-1 scoring
- **Visualizing transit access, sidewalk quality, amenity proximity, and street connectivity**
- **Identifying infrastructure gaps and equity concerns** across census tracts
- **Providing actionable insights** for urban planners and policymakers
- **Tracking data lineage and quality** for reproducible analysis

## 🚀 Current Status

### ✅ **Completed Phases:**
- **Phase 1-4**: Data acquisition and processing pipeline
- **Phase 5**: Testing and validation (100% complete)
- **Phase 6**: Visualization updates (100% complete)

### 🎉 **Production Ready Features:**
- **Real-time data processing** from GTFS, Census, and OSM sources
- **Interactive geospatial dashboard** with Streamlit
- **Comprehensive accessibility metrics** (95.6% wheelchair accessible stops)
- **Multi-layer visualization** of mobility metrics
- **Neighborhood-level accessibility scoring** across 495 census tracts
- **Data quality assessment** and lineage tracking
- **Robust error handling** and user feedback

## 📊 Key Metrics

### **Current Data Coverage:**
- **Transit Data**: 6,836 stops across 2 agencies
- **Census Data**: 495 tracts covering 2.24M people
- **OSM Data**: 162,972 sidewalk segments, 3,004 amenities
- **Mobility Index**: Complete coverage with normalized 0-1 scoring

### **Accessibility Analysis:**
- **Wheelchair Accessible Stops**: 6,533 (95.6%)
- **Average Service Frequency**: 67 trips/day per stop
- **Average Headway**: 27.1 minutes
- **High Access Tracts**: 13 (2.6% of total)

## 🛠️ Features

### **Data Processing:**
- **Automated GTFS processing** with service frequency calculation
- **Census data integration** with demographic analysis
- **OSM infrastructure extraction** (sidewalks, amenities, street networks)
- **Quality assessment** and validation
- **Data lineage tracking** for reproducibility

### **Visualization:**
- **Interactive transit maps** with accessibility color coding
- **Mobility index choropleth** by census tract
- **Real-time metrics dashboard** with key statistics
- **Multi-tab analysis** (transit, sidewalks, amenities, mobility index)
- **Responsive design** for different screen sizes

### **Analysis:**
- **Mobility Accessibility Index** combining transit, sidewalk, and amenity scores
- **Equity gap identification** across neighborhoods
- **Service coverage analysis** by transit agency
- **Infrastructure quality assessment**

## 📁 Data Sources

- **Census API**: Demographic and commuting data (ACS 5-year estimates)
- **GTFS feeds**: Real-time transit data from King County Metro and Sound Transit
- **OpenStreetMap**: Infrastructure data (sidewalks, amenities, street networks)
- **Local government datasets**: Additional context and validation

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/swarnimaj/urban-mobility-analytics.git
cd urban-mobility-analytics
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run src/visualization/app.py
```

The app will be available at `http://localhost:8501`

## 📂 Project Structure

```
urban-mobility-analytics/
├── data/                          # Data files
│   ├── raw/                       # Original, immutable data
│   ├── interim/                   # Intermediate processed data
│   └── processed/                 # Final, analysis-ready data
│       ├── census/                # Census tract data
│       ├── transit/               # GTFS processed data
│       ├── osm/                   # OpenStreetMap data
│       ├── integrated/            # Combined mobility index
│       ├── metadata.json          # Dataset metadata
│       ├── data_lineage.json      # Data lineage tracking
│       └── quality_report.json    # Quality assessment
├── src/                           # Source code
│   ├── data_acquisition/          # Data collection and processing
│   ├── utils/                     # Utility functions and pipeline
│   └── visualization/             # Streamlit app and visualization
├── docs/                          # Documentation
├── tests/                         # Test cases
└── requirements.txt               # Python dependencies
```

## 🔧 Configuration

### **Environment Variables:**
- `CENSUS_API_KEY`: Required for Census data access
- `DEFAULT_CITY`: Default city for analysis (default: "Seattle")
- `DEFAULT_STATE`: Default state for analysis (default: "WA")

### **Data Processing:**
- **GTFS feeds** are automatically downloaded and processed
- **Census data** is fetched via API with caching
- **OSM data** is extracted using bounding boxes
- **Quality assessment** runs automatically on all datasets

## 📈 Usage Examples

### **Running the Complete Pipeline:**
```python
from src.utils.pipeline_runner import PipelineRunner

# Run complete pipeline for Seattle
PipelineRunner.run_pipeline("Seattle")
```

### **Loading and Analyzing Data:**
```python
from src.visualization.app import load_transit_data, load_mobility_data

# Load latest data
transit_data = load_transit_data()
mobility_data = load_mobility_data()

# Analyze accessibility
accessible_stops = transit_data[transit_data['wheelchair_accessible'] == 'yes']
print(f"Accessibility rate: {len(accessible_stops)/len(transit_data)*100:.1f}%")
```

## 🧪 Testing

### **Run All Tests:**
```bash
python -m pytest tests/
```

### **Test Specific Components:**
```bash
# Test data loading
python test_app_data_loading.py

# Test visualization features
python test_visualization_features.py

# Test complete pipeline
python test_complete_pipeline.py
```

## 📊 Data Quality

### **Quality Metrics:**
- **Data completeness**: >95% for all major datasets
- **Accessibility coverage**: 95.6% of transit stops have accessibility data
- **Spatial accuracy**: All data properly geocoded and validated
- **Temporal relevance**: Latest available data from all sources

### **Validation:**
- **Automated quality checks** on all processed data
- **Data lineage tracking** for full traceability
- **Metadata generation** with comprehensive dataset information
- **Error reporting** and handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **King County Metro** and **Sound Transit** for GTFS data
- **U.S. Census Bureau** for demographic data
- **OpenStreetMap contributors** for infrastructure data
- **Streamlit** for the interactive dashboard framework

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**🎉 The Urban Mobility Analytics Platform is production-ready and actively providing insights into urban mobility equity!**
