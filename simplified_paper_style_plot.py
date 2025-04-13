#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to create a clean violin plot from saved slope data,
matching the paper's convention by taking absolute values of renewable slopes.

This version maintains the clean, paper-like appearance of the plot.

Usage:
    python simplified_paper_style_plot.py [input_file] [output_file]
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def create_violin_plot(slopes_data, output_path):
    """
    Create a violin plot of SHAP dependency slopes for selected features,
    matching the paper's convention by taking the absolute value of renewable slopes.
    """
    # Define the features to include and their display names
    features = ['load_forecast', 'solar_forecast', 'wind_forecast']
    feature_display_names = {
        'load_forecast': 'Load day-ahead',
        'solar_forecast': 'Solar day-ahead',
        'wind_forecast': 'Wind day-ahead'
    }
    
    # Process slopes to match paper convention
    feature_values = []
    for feature in features:
        if feature in slopes_data:
            # Take absolute value for renewable generation features
            if feature in ['solar_forecast', 'wind_forecast']:
                values = np.abs(slopes_data[feature])
                print(f"Taking absolute value of {feature} slopes to match paper convention")
            else:
                values = slopes_data[feature]
            feature_values.append(values)
        else:
            print(f"Warning: Feature '{feature}' not found in the data")
            sys.exit(1)
    
    # Set colors to match the reference image
    colors = ['#D5F0F2', '#65B1C1', '#3D629B']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create violin plot with clean styling
    positions = [1, 2, 3]
    violin_parts = plt.violinplot(
        feature_values, 
        positions=positions,
        showmeans=False, 
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
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
    plt.title('d', loc='left', fontweight='bold', fontsize=14)
    plt.xticks(positions, [feature_display_names[feature] for feature in features])
    plt.ylabel('Slope [EUR MWh$^{-2}$]')
    
    # Add grid lines (y-axis only)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Set y-limits to match the paper's plot
    plt.ylim(0.0004, 0.0014)
    
    # Save the plot
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
        output_file = os.path.join(PLOT_DIR, "paper_style_slopes.png")
    
    # Load the data
    slopes_data = load_slopes_data(input_file)
    
    # Create the violin plot
    final_path = create_violin_plot(slopes_data, output_file)
    
    print(f"\nViolin plot with paper's convention saved to: {final_path}")
    print("\nThis plot uses absolute values for renewable generation slopes to match")
    print("the paper's convention of 'multiplying the renewable generations by -1'.")

if __name__ == "__main__":
    main()