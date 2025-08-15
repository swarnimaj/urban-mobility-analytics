# src/utils/pipeline_runner.py
"""
Pipeline runner for the urban mobility project.

This module orchestrates the entire data processing workflow:
- Fetches data from multiple sources (Census, Transit, OSM)
- Integrates and processes the data
- Creates visualizations
- Tracks progress and handles errors

It can run the full pipeline or individual steps as needed.
"""

import os
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import time

# Import project modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from .config_manager import ConfigManager
from .data_cleaner import DataCleaner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Orchestrates the data processing pipeline."""
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.cleaner = DataCleaner(config_path)
        self.run_stats = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'steps_completed': [],
            'errors': []
        }
    
    def run_pipeline(self, city_name, steps=None, force_refresh=False):
        """
        Run the data processing pipeline.
        
        Args:
            city_name: Name of the city to process
            steps: List of steps to run (default: all)
            force_refresh: Whether to force refresh all data
            
        Returns:
            Dictionary with pipeline results
        """
        # Record start time
        self.run_stats['start_time'] = datetime.now().isoformat()
        
        # Get city configuration
        city_config = self.config.get_city_config(city_name)
        if not city_config:
            error_msg = f"City configuration not found for {city_name}"
            logger.error(error_msg)
            self.run_stats['errors'].append(error_msg)
            return self._finish_pipeline()
        
        # Define pipeline steps
        all_steps = [
            'fetch_census',
            'fetch_transit', 
            'fetch_osm',
            'integrate_data',
            'create_visualizations'
        ]
        
        steps_to_run = steps if steps else all_steps
        
        try:
            # Run each step
            for step in steps_to_run:
                if step in all_steps:
                    self._run_step(step, city_name, city_config, force_refresh)
                else:
                    logger.warning(f"Unknown step: {step}")
            
            # Save data lineage
            self.cleaner.save_data_lineage()
            
            return self._finish_pipeline()
            
        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            logger.error(error_msg)
            self.run_stats['errors'].append(error_msg)
            return self._finish_pipeline()
    
    def _run_step(self, step, city_name, city_config, force_refresh):
        """
        Run a single pipeline step.
        
        Args:
            step: Step name to run
            city_name: Name of the city
            city_config: City configuration
            force_refresh: Whether to force refresh data
        """
        step_info = {
            'fetch_census': {
                'name': 'Fetching Census data',
                'func': self.cleaner.fetch_census_data,
                'args': {'state': city_config['state'], 'county': city_config['county']}
            },
            'fetch_transit': {
                'name': 'Fetching transit data',
                'func': self.cleaner.fetch_transit_data,
                'args': {'city_name': city_name}
            },
            'fetch_osm': {
                'name': 'Fetching OSM data',
                'func': self.cleaner.fetch_osm_data,
                'args': {'place_name': f"{city_name}, {city_config['state']}"}
            },
            'integrate_data': {
                'name': 'Integrating data',
                'func': self.cleaner.integrate_data,
                'args': {'city_name': city_name}
            },
            'create_visualizations': {
                'name': 'Creating visualizations',
                'func': self.cleaner.create_basic_visualization,
                'args': {}
            }
        }
        
        if step not in step_info:
            return
        
        info = step_info[step]
        logger.info(f"Running: {info['name']}")
        
        start_time = time.time()
        
        try:
            # Run the step
            if step == 'integrate_data':
                result = info['func'](**info['args'])
                success = result is not None
            elif step == 'create_visualizations':
                result = info['func'](**info['args'])
                success = len(result) > 0
            else:
                success = info['func'](force_refresh=force_refresh, **info['args'])
            
            duration = time.time() - start_time
            
            if success:
                # Record successful step
                step_record = {
                    'step': step,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add outputs for visualization step
                if step == 'create_visualizations' and result:
                    step_record['outputs'] = [str(p) for p in result]
                
                self.run_stats['steps_completed'].append(step_record)
                logger.info(f"{info['name']} completed successfully in {duration:.2f} seconds")
            else:
                error_msg = f"Failed to {info['name'].lower()}"
                logger.error(error_msg)
                self.run_stats['errors'].append(error_msg)
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error in {info['name'].lower()}: {e}"
            logger.error(error_msg)
            self.run_stats['errors'].append(error_msg)
    
    def _finish_pipeline(self):
        """
        Finish the pipeline and return results.
        
        Returns:
            Dictionary with pipeline results
        """
        # Calculate duration
        self.run_stats['end_time'] = datetime.now().isoformat()
        
        if self.run_stats['start_time']:
            start_time = datetime.fromisoformat(self.run_stats['start_time'])
            end_time = datetime.fromisoformat(self.run_stats['end_time'])
            duration = (end_time - start_time).total_seconds()
            self.run_stats['duration'] = duration
        
        # Save run statistics
        self._save_run_stats()
        
        # Return results
        return {
            'success': len(self.run_stats['errors']) == 0,
            'steps_completed': len(self.run_stats['steps_completed']),
            'errors': len(self.run_stats['errors']),
            'duration': self.run_stats.get('duration'),
            'details': self.run_stats
        }
    
    def _save_run_stats(self):
        """Save pipeline run statistics to file."""
        try:
            stats_dir = self.config.processed_dir / "pipeline_stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_file = stats_dir / f"pipeline_run_{timestamp}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(self.run_stats, f, indent=2)
            
            logger.info(f"Pipeline stats saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline stats: {e}")

def main():
    """Run the pipeline from command line."""
    parser = argparse.ArgumentParser(description='Run the Urban Mobility Analytics pipeline')
    parser.add_argument('--city', required=True, help='City name to process')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', help='Steps to run (default: all)')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh all data')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    runner = PipelineRunner(args.config)
    results = runner.run_pipeline(args.city, args.steps, args.force_refresh)
    
    # Print summary
    print("\nPipeline Run Summary:")
    print(f"Success: {'Yes' if results['success'] else 'No'}")
    print(f"Steps completed: {results['steps_completed']}")
    print(f"Errors: {results['errors']}")
    print(f"Duration: {results['duration']:.2f} seconds" if results['duration'] else "Duration: N/A")
    
    if results['errors'] > 0:
        print("\nErrors:")
        for error in results['details']['errors']:
            print(f"- {error}")
    
    return 0 if results['success'] else 1

if __name__ == "__main__":
    exit(main())