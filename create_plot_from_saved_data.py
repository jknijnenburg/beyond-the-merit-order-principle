#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a violin plot from saved slope data.

This script loads previously calculated SHAP dependency slopes from a pickle file
and creates a violin plot with only selected features (load, solar, wind).

Usage:
    python create_plot_from_saved_data.py [input_file] [output_file]
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

# Set style for the plot
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']

# Default directories
PLOT_DIR = "./plots/"
CHECKPOINT_DIR = "./checkpoints/"
os.makedirs(PLOT_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create violin plot from saved slope data")
    parser.add_argument("input_file", nargs="?", default=None,
                        help="Path to the pickle file containing slope data")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="Path to save the output plot")
    parser.add_argument("--features", nargs="+", 
                        default=["load_forecast", "solar_forecast", "wind_forecast"],
                        help="Features to include in the plot")
    return parser.parse_args()

def find_latest_slopes_file():
    """Find the most recent slopes data file in the checkpoints directory"""
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("slopes_data_") and f.endswith(".pkl")]
    if not files:
        return None
    
    # Sort by modification time, newest first
    files.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    return os.path.join(CHECKPOINT_DIR, files[0])

def load_slopes_data(file_path):
    """Load slopes data from a pickle file"""
    try:
        with open(file_path, "rb") as f:
            slopes_data = pickle.load(f)
        print(f"Loaded slopes data from {file_path}")
        print(f"Available features: {list(slopes_data.keys())}")
        return slopes_data
    except Exception as e:
        print(f"Error loading slopes data: {str(e)}")
        sys.exit(1)

def create_violin_plot(slopes_data, features, output_path):
    """
    Create a violin plot of SHAP dependency slopes for selected features.
    
    Arguments:
        slopes_data: Dictionary mapping feature names to arrays of slope values
        features: List of features to include in the plot
        output_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    # Filter for only the requested features
    filtered_slopes = {}
    for feature in features:
        if feature in slopes_data:
            filtered_slopes[feature] = slopes_data[feature]
        else:
            print(f"Warning: Feature '{feature}' not found in the data")
    
    if not filtered_slopes:
        print("No valid features found. Please check the feature names.")
        sys.exit(1)
    
    # Map internal feature names to display names
    feature_display_names = {
        'load_forecast': 'Load day-ahead',
        'solar_forecast': 'Solar day-ahead',
        'wind_forecast': 'Wind day-ahead',
        'total_generation': 'Total generation',
        'net_import_export': 'Net import/export',
        'oil': 'Oil price',
        'natural_gas': 'Natural gas price'
    }
    
    # Get feature names and corresponding values
    feature_names = list(filtered_slopes.keys())
    feature_values = [filtered_slopes[name] for name in feature_names]
    positions = list(range(1, len(feature_names) + 1))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Set colors to match the reference image
    colors = ['#D5F0F2', '#65B1C1', '#3D629B']
    
    # Create violin plot
    violin_parts = plt.violinplot(
        feature_values,
        positions=positions,
        showmeans=False, 
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    # Add box plots inside the violins
    plt.boxplot(
        feature_values,
        positions=positions,
        widths=0.15,
        patch_artist=False,
        boxprops=dict(linestyle='-', linewidth=1.5, color='black'),
        whiskerprops=dict(linestyle='-', linewidth=1.5, color='black'),
        medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
        capprops=dict(linestyle='-', linewidth=1.5, color='black'),
        flierprops=dict(marker='.', markerfacecolor='black', markersize=3)
    )
    
    # Customize the plot
    plt.xticks(positions, [feature_display_names.get(name, name) for name in feature_names], fontsize=12)
    plt.ylabel('Slope [EUR MWh$^{-2}$]', fontsize=12)
    plt.title('d', loc='left', fontweight='bold', fontsize=14)
    
    # Add grid lines
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Lighten the border
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#888888')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Main function to create the violin plot from saved data"""
    args = parse_arguments()
    
    # Determine input file
    input_file = args.input_file
    if input_file is None:
        input_file = find_latest_slopes_file()
        if input_file is None:
            print("No slopes data file found. Please specify an input file.")
            sys.exit(1)
    
    # Determine output file
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(PLOT_DIR, "custom_shap_dependency_slopes.png")
    
    # Load the data
    slopes_data = load_slopes_data(input_file)
    
    # Print summary statistics for available features
    print("\nSlope summary statistics:")
    for feature in args.features:
        if feature in slopes_data:
            slopes = slopes_data[feature]
            print(f"  {feature}: count={len(slopes)}, mean={np.mean(slopes):.6f}, "
                f"std={np.std(slopes):.6f}, min={np.min(slopes):.6f}, max={np.max(slopes):.6f}")
        else:
            print(f"  {feature}: Not available in the data")
    
    # Create the violin plot
    final_path = create_violin_plot(slopes_data, args.features, output_file)
    
    print(f"\nViolin plot saved to: {final_path}")

if __name__ == "__main__":
    main()