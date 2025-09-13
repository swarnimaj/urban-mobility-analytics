# src/visualization/transit_visualizations.py
"""
Transit Access Visualization Functions for Urban Mobility Analytics.

This module provides visualization functions for transit accessibility analysis,
including maps, charts, and comparative visualizations.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Optional import for enhanced plotting
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("Seaborn not available. Using default matplotlib styling.")

# Optional imports for enhanced visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some interactive visualizations will be disabled.")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    warnings.warn("Folium not available. Interactive maps will be disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Visualization configuration
plt.style.use('default')

class TransitVisualizationSuite:
    """
    Comprehensive transit accessibility visualization suite.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the visualization suite.
        
        Args:
            figsize: Default figure size for matplotlib plots
            dpi: DPI for high-quality output
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_schemes = {
            'accessibility': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850'],
            'distance': ['#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#ffff8c', '#f9d057', '#f29e2e', '#e76818'],
            'frequency': ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#d9f0d3', '#a6dba0', '#5aae61', '#1b7837']
        }
    
    def create_transit_access_map(self, 
                                neighborhoods: gpd.GeoDataFrame,
                                transit_stops: gpd.GeoDataFrame,
                                score_column: str = 'transit_access_score',
                                output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a choropleth map of transit accessibility scores.
        
        Args:
            neighborhoods: GeoDataFrame with transit scores
            transit_stops: GeoDataFrame of transit stops
            score_column: Column name for transit scores
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            logger.info("Creating transit access map")
            
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            
            # Plot neighborhoods colored by transit score
            if score_column in neighborhoods.columns:
                neighborhoods.plot(
                    column=score_column,
                    ax=ax,
                    legend=True,
                    cmap='RdYlGn',
                    edgecolor='white',
                    linewidth=0.5,
                    legend_kwds={
                        'label': 'Transit Access Score',
                        'shrink': 0.8,
                        'orientation': 'vertical'
                    }
                )
            else:
                logger.warning(f"Score column '{score_column}' not found, using default styling")
                neighborhoods.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
            
            # Plot transit stops
            if not transit_stops.empty:
                # Color by accessibility if available
                if 'wheelchair_accessible' in transit_stops.columns:
                    colors = []
                    for _, stop in transit_stops.iterrows():
                        if stop['wheelchair_accessible'] == 'yes':
                            colors.append('green')
                        elif stop['wheelchair_accessible'] == 'no':
                            colors.append('red')
                        else:
                            colors.append('blue')
                    
                    transit_stops.plot(ax=ax, color=colors, markersize=20, alpha=0.7)
                    
                    # Create custom legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                               markersize=8, label='Wheelchair Accessible'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                               markersize=8, label='Not Accessible'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                               markersize=8, label='Unknown')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                else:
                    transit_stops.plot(ax=ax, color='blue', markersize=15, alpha=0.7)
            
            # Styling
            ax.set_title('Transit Accessibility by Neighborhood', fontsize=16, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Transit access map saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating transit access map: {e}")
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, f'Error creating map:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_score_distribution_plot(self, 
                                     neighborhoods: gpd.GeoDataFrame,
                                     score_columns: Optional[List[str]] = None,
                                     output_path: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for transit access scores.
        
        Args:
            neighborhoods: GeoDataFrame with transit scores
            score_columns: List of score columns to plot
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            logger.info("Creating score distribution plots")
            
            if score_columns is None:
                # Auto-detect score columns
                score_columns = [col for col in neighborhoods.columns 
                               if 'score' in col.lower() and neighborhoods[col].dtype in ['float64', 'int64']]
            
            if not score_columns:
                logger.warning("No score columns found")
                fig, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.text(0.5, 0.5, 'No score columns found', ha='center', va='center')
                return fig
            
            # Create subplots
            n_cols = min(2, len(score_columns))
            n_rows = (len(score_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, score_col in enumerate(score_columns):
                if score_col in neighborhoods.columns:
                    data = neighborhoods[score_col].dropna()
                    
                    if len(data) > 0:
                        # Histogram with KDE
                        axes[i].hist(data, bins=30, density=True, alpha=0.7, 
                                   color=self.color_schemes['accessibility'][i % len(self.color_schemes['accessibility'])])
                        
                        # Add KDE curve (with error handling for singular matrices)
                        try:
                            from scipy.stats import gaussian_kde
                            # Check if data has sufficient variance for KDE
                            if len(data.unique()) > 1 and data.std() > 0:
                                kde = gaussian_kde(data)
                                x_range = np.linspace(data.min(), data.max(), 100)
                                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2)
                        except (ImportError, np.linalg.LinAlgError, ValueError) as e:
                            # Skip KDE if there are issues (singular matrix, insufficient variance, etc.)
                            logger.warning(f"Skipping KDE for {score_col}: {e}")
                            pass
                        
                        # Add statistics
                        mean_val = data.mean()
                        median_val = data.median()
                        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
                        axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
                        
                        axes[i].set_title(f'Distribution of {score_col.replace("_", " ").title()}', fontweight='bold')
                        axes[i].set_xlabel('Score')
                        axes[i].set_ylabel('Density')
                        axes[i].legend()
                        axes[i].grid(True, alpha=0.3)
                    else:
                        axes[i].text(0.5, 0.5, f'No data for {score_col}', ha='center', va='center')
                else:
                    axes[i].text(0.5, 0.5, f'Column {score_col} not found', ha='center', va='center')
            
            # Hide unused subplots
            for i in range(len(score_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Score distribution plots saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating score distribution plots: {e}")
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, f'Error creating plots:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_distance_frequency_scatter(self, 
                                        neighborhoods: gpd.GeoDataFrame,
                                        distance_col: str = 'nearest_stop_distance',
                                        frequency_col: str = 'frequency_score',
                                        score_col: str = 'transit_access_score',
                                        output_path: Optional[str] = None) -> plt.Figure:
        """
        Create scatter plot of distance vs frequency colored by transit score.
        
        Args:
            neighborhoods: GeoDataFrame with transit data
            distance_col: Column name for distance data
            frequency_col: Column name for frequency data
            score_col: Column name for transit scores
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            logger.info("Creating distance-frequency scatter plot")
            
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
            
            # Prepare data
            plot_data = neighborhoods[[distance_col, frequency_col, score_col]].dropna()
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No valid data for scatter plot', ha='center', va='center')
                return fig
            
            # Create scatter plot
            scatter = ax.scatter(
                plot_data[distance_col], 
                plot_data[frequency_col],
                c=plot_data[score_col],
                cmap='RdYlGn',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Transit Access Score', rotation=270, labelpad=20)
            
            # Styling
            ax.set_xlabel(f'{distance_col.replace("_", " ").title()} (meters)', fontsize=12)
            ax.set_ylabel(f'{frequency_col.replace("_", " ").title()}', fontsize=12)
            ax.set_title('Transit Access: Distance vs Frequency', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if possible
            try:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(
                    plot_data[distance_col], plot_data[frequency_col]
                )
                
                x_trend = np.array([plot_data[distance_col].min(), plot_data[distance_col].max()])
                y_trend = slope * x_trend + intercept
                
                ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2,
                       label=f'R² = {r_value**2:.3f}')
                ax.legend()
                
            except ImportError:
                pass
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Distance-frequency scatter plot saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_comparative_analysis(self, 
                                  neighborhoods: gpd.GeoDataFrame,
                                  grouping_col: str = 'accessibility_category',
                                  score_col: str = 'transit_access_score',
                                  output_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparative analysis plots (box plots, bar charts).
        
        Args:
            neighborhoods: GeoDataFrame with transit data
            grouping_col: Column to group by for comparison
            score_col: Column with scores to compare
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            logger.info("Creating comparative analysis plots")
            
            fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]), dpi=self.dpi)
            
            # Prepare data
            plot_data = neighborhoods[[grouping_col, score_col]].dropna()
            
            if len(plot_data) == 0:
                for ax in axes:
                    ax.text(0.5, 0.5, 'No valid data for comparison', ha='center', va='center')
                return fig
            
            # Box plot
            if len(plot_data[grouping_col].unique()) > 1:
                plot_data.boxplot(column=score_col, by=grouping_col, ax=axes[0])
                axes[0].set_title(f'{score_col.replace("_", " ").title()} by {grouping_col.replace("_", " ").title()}')
                axes[0].set_xlabel(grouping_col.replace("_", " ").title())
                axes[0].set_ylabel(score_col.replace("_", " ").title())
                
                # Bar chart with means
                group_means = plot_data.groupby(grouping_col)[score_col].mean()
                group_means.plot(kind='bar', ax=axes[1], color=self.color_schemes['accessibility'][:len(group_means)])
                axes[1].set_title(f'Average {score_col.replace("_", " ").title()} by Group')
                axes[1].set_xlabel(grouping_col.replace("_", " ").title())
                axes[1].set_ylabel(f'Average {score_col.replace("_", " ").title()}')
                axes[1].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(group_means.values):
                    axes[1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
            else:
                for ax in axes:
                    ax.text(0.5, 0.5, f'Only one group found in {grouping_col}', ha='center', va='center')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Comparative analysis plots saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparative analysis: {e}")
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, f'Error creating analysis:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def create_interactive_map(self, 
                             neighborhoods: gpd.GeoDataFrame,
                             transit_stops: gpd.GeoDataFrame,
                             score_column: str = 'transit_access_score',
                             output_path: Optional[str] = None) -> Optional[str]:
        """
        Create interactive folium map of transit accessibility.
        
        Args:
            neighborhoods: GeoDataFrame with transit scores
            transit_stops: GeoDataFrame of transit stops
            score_column: Column name for transit scores
            output_path: Optional path to save the HTML map
            
        Returns:
            Path to saved HTML file or None if folium not available
        """
        if not FOLIUM_AVAILABLE:
            logger.warning("Folium not available, skipping interactive map")
            return None
        
        try:
            logger.info("Creating interactive transit access map")
            
            # Calculate map center
            bounds = neighborhoods.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add neighborhoods choropleth
            if score_column in neighborhoods.columns:
                folium.Choropleth(
                    geo_data=neighborhoods,
                    data=neighborhoods,
                    columns=['geoid' if 'geoid' in neighborhoods.columns else neighborhoods.index, score_column],
                    key_on='feature.properties.geoid' if 'geoid' in neighborhoods.columns else 'feature.id',
                    fill_color='RdYlGn',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name='Transit Access Score'
                ).add_to(m)
            
            # Add transit stops
            for idx, stop in transit_stops.iterrows():
                # Determine color based on accessibility
                if 'wheelchair_accessible' in stop:
                    if stop['wheelchair_accessible'] == 'yes':
                        color = 'green'
                        icon = 'ok-sign'
                    elif stop['wheelchair_accessible'] == 'no':
                        color = 'red'
                        icon = 'remove-sign'
                    else:
                        color = 'blue'
                        icon = 'question-sign'
                else:
                    color = 'blue'
                    icon = 'info-sign'
                
                # Create popup text
                popup_text = f"<b>{stop.get('stop_name', 'Transit Stop')}</b><br>"
                popup_text += f"Stop ID: {stop.get('stop_id', 'N/A')}<br>"
                if 'wheelchair_accessible' in stop:
                    popup_text += f"Wheelchair Accessible: {stop['wheelchair_accessible']}<br>"
                if 'trips_per_day' in stop:
                    popup_text += f"Trips per Day: {stop['trips_per_day']}<br>"
                
                folium.Marker(
                    location=[stop.geometry.y, stop.geometry.x],
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            if output_path is None:
                output_path = 'transit_access_map.html'
            
            m.save(output_path)
            logger.info(f"Interactive map saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {e}")
            return None
    
    def create_comprehensive_dashboard(self, 
                                     neighborhoods: gpd.GeoDataFrame,
                                     transit_stops: gpd.GeoDataFrame,
                                     output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            neighborhoods: GeoDataFrame with transit scores
            transit_stops: GeoDataFrame of transit stops
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        try:
            logger.info("Creating comprehensive transit accessibility dashboard")
            
            if output_dir is None:
                output_dir = Path.cwd() / "transit_visualizations"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # 1. Transit access map
            try:
                fig = self.create_transit_access_map(neighborhoods, transit_stops)
                map_path = output_dir / "transit_access_map.png"
                fig.savefig(map_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files['access_map'] = str(map_path)
            except Exception as e:
                logger.error(f"Error creating access map: {e}")
            
            # 2. Score distributions
            try:
                fig = self.create_score_distribution_plot(neighborhoods)
                dist_path = output_dir / "score_distributions.png"
                fig.savefig(dist_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files['distributions'] = str(dist_path)
            except Exception as e:
                logger.error(f"Error creating distributions: {e}")
            
            # 3. Distance-frequency scatter
            try:
                fig = self.create_distance_frequency_scatter(neighborhoods)
                scatter_path = output_dir / "distance_frequency_scatter.png"
                fig.savefig(scatter_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files['scatter'] = str(scatter_path)
            except Exception as e:
                logger.error(f"Error creating scatter plot: {e}")
            
            # 4. Comparative analysis
            try:
                # Create accessibility categories if not present
                if 'accessibility_category' not in neighborhoods.columns and 'transit_access_score' in neighborhoods.columns:
                    neighborhoods['accessibility_category'] = pd.cut(
                        neighborhoods['transit_access_score'],
                        bins=[0, 25, 50, 75, 100],
                        labels=['Poor', 'Fair', 'Good', 'Excellent']
                    )
                
                fig = self.create_comparative_analysis(neighborhoods)
                comp_path = output_dir / "comparative_analysis.png"
                fig.savefig(comp_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files['comparative'] = str(comp_path)
            except Exception as e:
                logger.error(f"Error creating comparative analysis: {e}")
            
            # 5. Interactive map
            try:
                interactive_path = self.create_interactive_map(
                    neighborhoods, transit_stops, 
                    output_path=str(output_dir / "interactive_map.html")
                )
                if interactive_path:
                    saved_files['interactive_map'] = interactive_path
            except Exception as e:
                logger.error(f"Error creating interactive map: {e}")
            
            logger.info(f"Dashboard created with {len(saved_files)} visualizations in {output_dir}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return {}


# Convenience functions for easy use

def visualize_transit_access(neighborhoods: gpd.GeoDataFrame,
                           transit_stops: gpd.GeoDataFrame,
                           output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Create all transit access visualizations.
    
    Args:
        neighborhoods: GeoDataFrame with transit scores
        transit_stops: GeoDataFrame of transit stops
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of saved file paths
    """
    visualizer = TransitVisualizationSuite()
    return visualizer.create_comprehensive_dashboard(neighborhoods, transit_stops, output_dir)


def create_simple_transit_map(neighborhoods: gpd.GeoDataFrame,
                            transit_stops: gpd.GeoDataFrame,
                            score_column: str = 'transit_access_score') -> plt.Figure:
    """
    Create a simple transit access map.
    
    Args:
        neighborhoods: GeoDataFrame with transit scores
        transit_stops: GeoDataFrame of transit stops
        score_column: Column name for transit scores
        
    Returns:
        Matplotlib figure
    """
    visualizer = TransitVisualizationSuite()
    return visualizer.create_transit_access_map(neighborhoods, transit_stops, score_column)
