"""
Visualization Generator for Evaluation Data

This script generates visualizations from existing evaluation data directories.
It can create various types of plots based on different evaluation methods.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate visualizations from evaluation data')
    
    parser.add_argument('--data-dir', type=str, default="./logs",
                       help='Directory containing evaluation data')
    
    parser.add_argument('--output-dir', type=str, default="./plots",
                       help='Directory to save visualizations (default: ./plots)')
    
    parser.add_argument('--report-types', nargs='+', default=['all'],
                        choices=['all', 'convergence', 'fixed-episodes', 'fixed-resource', 'cnp'],
                        help='Types of reports to generate (default: all)')
                        
    parser.add_argument('--format', nargs='+', default=['png', 'pdf'],
                        choices=['png', 'pdf', 'svg'],
                        help='Output image formats (default: png, pdf)')
    
    return parser.parse_args()

def find_evaluation_files(data_dir):
    """
    Find evaluation files in the data directory
    
    Args:
        data_dir: Directory to search
        
    Returns:
        dict: Files organized by evaluation type
    """
    data_dir = Path(data_dir)
    evaluation_files = {
        'integrated': [],
        'convergence': [],
        'fixed_episodes': [],
        'fixed_resource': [],
        'cnp': []
    }
    
    # Find integrated evaluation reports
    for filepath in data_dir.glob('integrated_evaluation_report_*.json'):
        evaluation_files['integrated'].append(filepath)
    
    # Find convergence evaluation directories
    for dirpath in data_dir.glob('convergence_eval_*'):
        summary_file = dirpath / 'convergence_summary.json'
        if summary_file.exists():
            evaluation_files['convergence'].append(summary_file)
    
    # Find fixed episodes evaluation directories
    for dirpath in data_dir.glob('fixed_episodes_*'):
        summary_file = dirpath / 'evaluation_summary.json'
        if summary_file.exists():
            evaluation_files['fixed_episodes'].append(summary_file)
    
    # Find fixed resource evaluation directories
    for dirpath in data_dir.glob('fixed_resource_*'):
        summary_file = dirpath / 'evaluation_summary.json'
        if summary_file.exists():
            evaluation_files['fixed_resource'].append(summary_file)
    
    # Find CNP metrics directories
    for dirpath in data_dir.glob('cnp_metrics_*'):
        metrics_file = dirpath / 'cnp_metrics.json'
        if metrics_file.exists():
            evaluation_files['cnp'].append(metrics_file)
    
    return evaluation_files

def load_visualization_module():
    """
    Load the visualization module
    
    Returns:
        module or None: Loaded visualization module
    """
    try:
        from visualization import cnp_analysis
        return cnp_analysis
    except ImportError:
        print("Error: visualization.cnp_analysis module not found.")
        print("Please confirm installation of visualization module.")
        return None

def generate_visualizations(data_files, output_dir, report_types, formats):
    """
    Generate visualizations from evaluation data files
    
    Args:
        data_files: Dictionary of evaluation files
        output_dir: Output directory for visualizations
        report_types: Types of reports to generate
        formats: Output formats for visualizations
        
    Returns:
        bool: Success or failure
    """
    visualization_module = load_visualization_module()
    if not visualization_module:
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Call visualization module functions
    try:
        if 'all' in report_types or 'cnp' in report_types:
            if data_files['cnp']:
                latest_cnp_file = max(data_files['cnp'], key=os.path.getmtime)
                print(f"CNP metrics file: {latest_cnp_file}")
                visualization_module.run_cnp_analysis(str(latest_cnp_file.parent), output_dir, 
                                                    formats=formats)
                print(f"CNP visualizations saved to {output_dir}")
                
        # Implement other visualization types (for extensibility)
        if hasattr(visualization_module, 'run_convergence_analysis') and ('all' in report_types or 'convergence' in report_types):
            if data_files['convergence']:
                latest_conv_file = max(data_files['convergence'], key=os.path.getmtime)
                print(f"Convergence metrics file: {latest_conv_file}")
                visualization_module.run_convergence_analysis(str(latest_conv_file.parent), output_dir,
                                                            formats=formats)
                print(f"Convergence visualizations saved to {output_dir}")
                
        if hasattr(visualization_module, 'run_comparison_analysis') and ('all' in report_types):
            if data_files['integrated']:
                latest_integrated = max(data_files['integrated'], key=os.path.getmtime)
                print(f"Integrated metrics file: {latest_integrated}")
                visualization_module.run_comparison_analysis(str(latest_integrated), output_dir,
                                                          formats=formats)
                print(f"Comparison visualizations saved to {output_dir}")
                
        return True
        
    except AttributeError as e:
        print(f"Error: Visualization module missing required function: {e}")
        return False
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to coordinate visualization generation
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_args()
    
    # Verify data directory
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: Specified data directory does not exist: {data_dir}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir
    
    # Find evaluation files
    print(f"Searching data directory: {data_dir}")
    evaluation_files = find_evaluation_files(data_dir)
    
    # Check file count
    total_files = sum(len(files) for files in evaluation_files.values())
    if total_files == 0:
        print("Warning: No evaluation files found in the specified directory.")
        return 1
    
    print(f"Found {total_files} evaluation files:")
    for key, files in evaluation_files.items():
        print(f"  {key}: {len(files)} files")
    
    # Generate visualizations
    print(f"\nGenerating visualizations in output directory: {output_dir}")
    success = generate_visualizations(evaluation_files, output_dir, args.report_types, args.format)
    
    if success:
        print(f"\nVisualization generation complete. Output directory: {output_dir}")
        return 0
    else:
        print("\nErrors occurred during visualization generation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())