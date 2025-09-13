# Transit Access Score Methodology

## Overview

The Transit Access Score is a comprehensive metric that evaluates the accessibility and quality of public transit services for each neighborhood (census tract). This score combines multiple factors to provide a holistic assessment of transit accessibility, with particular attention to equity and accessibility for people with disabilities.

## Scoring Components

The Transit Access Score is calculated as a weighted combination of four key components:

### 1. Distance Score (40% weight)
**Purpose**: Measures how close residents are to transit stops.

**Calculation**:
- Calculates distance from neighborhood centroid to nearest transit stop
- Supports multiple distance calculation methods:
  - **Euclidean**: Straight-line distance (fastest)
  - **Walking Network**: Distance along pedestrian paths (most accurate)
  - **Multiple Stops**: Considers access to multiple nearby stops

**Scoring**:
- Maximum score (100) for stops within immediate vicinity
- Linear decay to 0 for stops beyond maximum walking distance (default: 1000m)
- Formula: `score = 100 * (1 - distance / max_distance)`

### 2. Service Frequency Score (30% weight)
**Purpose**: Evaluates the quality and frequency of transit service.

**Metrics Used**:
- **Trips per day**: Total daily service frequency
- **Average headway**: Time between consecutive transit vehicles
- **Service hours**: Span of daily service

**Calculation**:
- Aggregates service metrics for all stops within walking distance
- Weights stops by their proximity to the neighborhood
- Higher frequency and longer service hours result in higher scores

**Scoring**:
- Trips per day component (70%): `min(100, (total_trips_per_day / 100) * 10)`
- Headway component (30%): `max(0, 100 - avg_headway_minutes)`

### 3. Accessibility Score (20% weight)
**Purpose**: Assesses wheelchair accessibility and accommodation for people with disabilities.

**Metrics Used**:
- Proportion of nearby stops that are wheelchair accessible
- Presence of accessibility features (tactile paving, audio announcements, etc.)

**Calculation**:
- Counts wheelchair-accessible stops within walking distance
- Calculates percentage of accessible stops
- Formula: `score = (accessible_stops / total_nearby_stops) * 100`

**Categories**:
- **Fully Accessible** (100 points): All nearby stops are wheelchair accessible
- **Partially Accessible** (50-99 points): Some stops are accessible
- **Not Accessible** (0 points): No accessible stops nearby
- **Unknown** (50 points): Accessibility information not available

### 4. Coverage Score (10% weight)
**Purpose**: Measures the density and variety of transit options.

**Metrics Used**:
- Number of stops within walking distance
- Number of different transit routes/agencies
- Coverage of different buffer distances (200m, 400m, 800m, 1200m)

**Calculation**:
- Counts total stops within maximum walking distance
- Normalizes by maximum count across all neighborhoods
- Formula: `score = (stop_count_within_buffer / max_stop_count) * 100`

## Methodology Details

### Distance Calculation Methods

#### Euclidean Distance
- **Method**: Straight-line distance between points
- **Pros**: Fast computation, good approximation
- **Cons**: Doesn't account for street network or barriers
- **Use case**: Large-scale analysis, initial screening

#### Walking Network Distance
- **Method**: Shortest path along pedestrian-accessible streets
- **Pros**: Most accurate representation of actual walking distance
- **Cons**: Requires detailed street network data, computationally intensive
- **Implementation**: Uses network penalty factor (1.35x) on Euclidean distance when full network routing unavailable

#### Multiple Stops Access
- **Method**: Evaluates access to multiple stops within various buffer distances
- **Buffers**: 200m, 400m, 800m, 1200m from neighborhood centroid
- **Pros**: Captures redundancy and choice in transit options
- **Use case**: Dense urban areas with multiple transit options

### Score Normalization

All component scores are normalized to a 0-100 scale using configurable methods:

#### Min-Max Normalization (Default)
```
normalized_score = (score - min_value) / (max_value - min_value) * 100
```

#### Z-Score Normalization
```
normalized_score = ((score - mean) / std_dev + 3) / 6 * 100
```

#### Robust Normalization
```
normalized_score = ((score - median) / IQR + 2) / 4 * 100
```

### Final Score Calculation

The final Transit Access Score is calculated as:

```
Transit_Access_Score = (
    0.40 * Distance_Score +
    0.30 * Frequency_Score +
    0.20 * Accessibility_Score +
    0.10 * Coverage_Score
)
```

**Note**: Weights are configurable and can be adjusted based on local priorities.

## Implementation Details

### Data Requirements

#### Transit Stops Data (Required)
- **stop_id**: Unique identifier for each stop
- **stop_name**: Human-readable stop name
- **stop_lat, stop_lon**: Geographic coordinates
- **geometry**: Point geometry (automatically created)

#### Service Frequency Data (Optional)
- **trips_per_day**: Total daily trips serving the stop
- **trips_per_hour**: Average hourly trips
- **avg_headway_minutes**: Average time between vehicles

#### Accessibility Data (Optional)
- **wheelchair_accessible**: Wheelchair accessibility ('yes', 'no', 'unknown')
- **tactile_paving**: Presence of tactile paving
- **audio_announcements**: Audio accessibility features

#### Neighborhood Data (Required)
- **geoid**: Unique identifier for neighborhood/census tract
- **geometry**: Polygon or point geometry

### Configuration Options

#### TransitScoreCalculator Parameters
```python
TransitScoreCalculator(
    max_walking_distance=1000,  # Maximum walking distance in meters
    walking_speed=4.5,          # Average walking speed in km/h
    weights={                   # Component weights (must sum to 1.0)
        'distance': 0.4,
        'frequency': 0.3,
        'accessibility': 0.2,
        'coverage': 0.1
    }
)
```

#### Distance Calculation Options
- **method**: 'euclidean', 'walking_network', 'multiple_stops'
- **buffer_sizes**: List of buffer distances for multiple stops analysis
- **network_penalty_factor**: Multiplier for euclidean distance when network routing unavailable

### Performance Considerations

#### Computational Complexity
- **Euclidean**: O(n*m) where n=neighborhoods, m=stops
- **Network**: O(n*m*k) where k=network complexity
- **Multiple Stops**: O(n*m*b) where b=number of buffer sizes

#### Optimization Strategies
1. **Spatial Indexing**: Use spatial indexes for distance calculations
2. **Chunked Processing**: Process large datasets in chunks
3. **Caching**: Cache frequently used calculations
4. **Parallel Processing**: Use multiprocessing for independent calculations

## Validation and Quality Assurance

### Data Quality Checks
1. **Completeness**: Verify required fields are present
2. **Accuracy**: Check for reasonable coordinate values
3. **Consistency**: Ensure data types and formats are correct
4. **Temporal**: Validate service frequency data currency

### Score Validation
1. **Range Checks**: Ensure scores are within 0-100 range
2. **Distribution Analysis**: Check for reasonable score distributions
3. **Correlation Analysis**: Verify expected relationships between components
4. **Outlier Detection**: Identify and investigate extreme scores

### Sensitivity Analysis
- Test impact of different weight configurations
- Analyze sensitivity to missing data
- Evaluate robustness to data quality issues

## Usage Examples

### Basic Usage
```python
from src.analysis.transit_score import TransitScoreCalculator

# Initialize calculator
calculator = TransitScoreCalculator()

# Calculate comprehensive scores
results = calculator.calculate_comprehensive_transit_score(
    neighborhoods=census_tracts,
    transit_stops=gtfs_stops,
    service_data=service_frequency
)
```

### Advanced Configuration
```python
# Custom configuration
calculator = TransitScoreCalculator(
    max_walking_distance=800,  # Shorter walking distance
    weights={
        'distance': 0.5,       # Higher emphasis on distance
        'frequency': 0.3,
        'accessibility': 0.15,
        'coverage': 0.05
    }
)

# Multiple distance methods
euclidean_results = calculator.calculate_distance_to_transit(
    neighborhoods, stops, method='euclidean'
)

network_results = calculator.calculate_distance_to_transit(
    neighborhoods, stops, method='walking_network'
)
```

### Integration with Data Pipeline
```python
from src.utils.data_cleaner import DataCleaner

# The data cleaner automatically uses the new scoring system
cleaner = DataCleaner()
cleaner.fetch_all_data("Seattle")
integrated_data = cleaner.integrate_data("Seattle")

# Access transit scores
transit_scores = integrated_data['census_with_mobility']['transit_access_score']
```

## Interpretation Guide

### Score Ranges
- **90-100**: Excellent transit access
  - Multiple nearby stops
  - High service frequency
  - Full wheelchair accessibility
  - Short walking distances

- **70-89**: Good transit access
  - Nearby stops with good service
  - Most stops wheelchair accessible
  - Reasonable walking distances

- **50-69**: Fair transit access
  - Some nearby stops
  - Moderate service frequency
  - Mixed accessibility
  - Longer walking distances

- **25-49**: Poor transit access
  - Few nearby stops
  - Low service frequency
  - Limited accessibility
  - Long walking distances

- **0-24**: Very poor transit access
  - No nearby stops or very distant
  - Minimal service
  - Poor accessibility
  - Walking distance exceeds comfort zone

### Component Analysis

Use individual component scores to identify specific improvement areas:

- **Low Distance Score**: Need more stops or better spatial coverage
- **Low Frequency Score**: Need increased service frequency or extended hours
- **Low Accessibility Score**: Need wheelchair accessibility improvements
- **Low Coverage Score**: Need better stop distribution or more routes

## Future Enhancements

### Planned Improvements
1. **Network Routing**: Full integration with street network routing libraries
2. **Multi-modal Integration**: Include bike-share, ride-share connections
3. **Temporal Analysis**: Time-of-day and day-of-week variations
4. **Real-time Data**: Integration with real-time transit feeds
5. **Equity Weighting**: Adjust scores based on demographic factors

### Research Opportunities
1. **Machine Learning**: ML-based score prediction and optimization
2. **Behavioral Analysis**: Integration with actual ridership data
3. **Environmental Factors**: Weather and seasonal impact analysis
4. **Economic Analysis**: Cost-benefit analysis of improvements

## References

1. Litman, T. (2017). Evaluating Public Transit Benefits and Costs. Victoria Transport Policy Institute.
2. Welch, T. F. (2013). Equity in transport: The distribution of transit access in Toronto. Transport Policy, 30, 1-10.
3. Mavoa, S., et al. (2012). GIS based destination accessibility via public transit and walking in Auckland, New Zealand. Journal of Transport Geography, 20(1), 15-22.
4. GTFS Reference: https://developers.google.com/transit/gtfs/reference
5. OpenStreetMap Accessibility Tags: https://wiki.openstreetmap.org/wiki/Key:wheelchair
