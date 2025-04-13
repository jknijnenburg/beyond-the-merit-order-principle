#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create comprehensive SHAP dependency and interaction plots with TRUE interactions.

This fixed implementation ensures the interaction columns (3-5) in the grid plot
show actual interaction values on the y-axis, exactly like in the single interaction plot.
Uses plt.subplots() for improved layout management.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def create_fixed_interaction_grid(model, output_path="fixed_shap_interactions_grid.png", dpi=300):
    """
    Create comprehensive SHAP dependency and interaction grid with TRUE interaction values.
    
    Args:
        model: Trained ElectricityPriceModel with SHAP values calculated
        output_path: Path to save the output figure
        dpi: Resolution for saved figure
        
    Returns:
        Path to the saved figure
    """
    # Calculate SHAP interaction values if not already done
    if not hasattr(model, 'explainer'):
        model.explainer = shap.TreeExplainer(model.model)
        
    if not hasattr(model, 'shap_interaction_values'):
        print("Calculating SHAP interaction values (this may take some time)...")
        model.shap_interaction_values = model.explainer.shap_interaction_values(model.X_test)
        print("SHAP interaction values calculated.")
    
    # Get feature indices mapping
    feature_indices = {feature: list(model.X_test.columns).index(feature) 
                      for feature in model.X_test.columns}
    
    # Define key features to analyze
    main_features = [
        "load_forecast",
        "solar_forecast", 
        "wind_forecast",
        "net_import_export"
    ]
    
    # Define interaction features for each main feature
    interaction_features = {
        "load_forecast": ["wind_forecast", "solar_forecast", "natural_gas"],
        "solar_forecast": ["load_forecast", "oil", "natural_gas"],
        "wind_forecast": ["load_forecast", "oil", "natural_gas"],
        "net_import_export": ["wind_forecast", "load_forecast", "natural_gas"]
    }
    
    # Display names for features
    feature_display_names = {
        "load_forecast": "Load day-ahead [MWh]",
        "solar_forecast": "Solar day-ahead [MWh]",
        "wind_forecast": "Wind day-ahead [MWh]",
        "net_import_export": "Import export day-ahead [MWh]",
        "oil": "Oil price",
        "natural_gas": "Natural gas price"
    }
    
    # Create figure with subplots
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 14), constrained_layout=True)
    
    # Row labels
    row_labels = ['a', 'b', 'c', 'd']
    
    # Add column headers
    fig.text(0.17, 0.98, "With all Interactions", ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.38, 0.98, "Without Interactions", ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.7, 0.98, "Interactions with three selected other Features", ha='center', va='center', fontsize=12, fontweight='bold')

    # Custom blue colormap like the paper
    colors = ["#000080", "#0000ff", "#1e90ff", "#87cefa", "#b0e0e6", "#e0ffff", "#ffffff"]
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    # Function to create plots for columns 1-2 (regular SHAP values)
    def create_regular_plot(ax, feature_idx, interaction_idx=None):
        """Create a regular SHAP dependence plot (columns 1-2)"""
        feature_values = model.X_test.iloc[:, feature_idx].values
        shap_values = model.shap_values[:, feature_idx]
        
        if interaction_idx is not None:
            # With interaction coloring (column 1)
            interaction_values = model.X_test.iloc[:, interaction_idx].values
            scatter = ax.scatter(feature_values, shap_values, 
                                c=interaction_values, cmap=cmap, 
                                s=5, alpha=0.8)
        else:
            # Without interaction coloring (column 2)
            scatter = ax.scatter(feature_values, shap_values, 
                                color="#1F77B4", s=5, alpha=0.8)
        
        ax.set_xlabel(feature_display_names[model.X_test.columns[feature_idx]])
        ax.grid(True, linestyle='--', alpha=0.3)
        return scatter
    
    # Function to create true interaction plots (columns 3-5)
    def create_interaction_plot(ax, main_idx, interaction_idx):
        """Create a TRUE interaction plot showing interaction components only (columns 3-5)"""
        feature_values = model.X_test.iloc[:, main_idx].values
    
        # Extract interaction values (SHAP stores interactions symmetrically)
        interaction_values = model.shap_interaction_values[:, main_idx, interaction_idx] + \
                            model.shap_interaction_values[:, interaction_idx, main_idx]
        
        # Get the coloring feature values
        color_values = model.X_test.iloc[:, interaction_idx].values
        
        # Create scatter plot
        scatter = ax.scatter(feature_values, interaction_values, 
                            c=color_values, cmap=cmap, 
                            s=5, alpha=0.8)
        
        ax.set_xlabel(feature_display_names[model.X_test.columns[main_idx]])
        ax.grid(True, linestyle='--', alpha=0.3)
        return scatter
    
    # Process each main feature (rows)
    for row, main_feature in enumerate(main_features):
        main_idx = feature_indices[main_feature]
        
        # Add row label
        axs[row, 0].text(-0.3, 0.5, row_labels[row], transform=axs[row, 0].transAxes,
                fontsize=12, fontweight='bold', va='center')
                
        # Column 1: Plot with all interactions (using strongest interaction for coloring)
        strongest_interaction = interaction_features[main_feature][0]
        interact_idx = feature_indices[strongest_interaction]
        scatter = create_regular_plot(axs[row, 0], main_idx, interact_idx)
        
        # Only add y-label for the first column
        axs[row, 0].set_ylabel("SHAP values\n(Electricity Price) [EUR/MWh]")
        
        # Column 2: Plot without interactions
        create_regular_plot(axs[row, 1], main_idx, None)
        
        # Columns 3-5: TRUE interaction plots with specific features
        for col, interaction_feature in enumerate(interaction_features[main_feature]):
            interact_idx = feature_indices[interaction_feature]
            
            # Add title for each cell based on the interaction feature
            axs[row, col+2].set_title(feature_display_names[interaction_feature], fontsize=10)
            
            # Create the TRUE interaction plot
            scatter = create_interaction_plot(axs[row, col+2], main_idx, interact_idx)
            
            # Add y-label only for the first column's interaction plots
            if col == 0 and row == 0:
                # Add a label for the interaction columns section (just once)
                axs[row, 2].text(-0.8, 1.1, "SHAP interaction values [EUR/MWh]", 
                            transform=axs[row, 2].transAxes, 
                            fontsize=10, va='center', rotation=90)
    
    # Remove y-labels for columns 2-5
    for row in range(len(main_features)):
        for col in range(1, 5):
            axs[row, col].set_ylabel("")
    
    # Add a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),
        cax=cbar_ax
    )
    cbar.set_label('Value of Interacting Feature')
    
    # Add Low-High labels to the colorbar
    cbar_ax.text(0.5, 0.01, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes)
    cbar_ax.text(0.5, 0.99, 'High', ha='center', va='top', transform=cbar_ax.transAxes)
    
    # Adjust layout
    plt.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.05, wspace=0.5, hspace=0.5)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed SHAP interaction grid saved to {output_path}")
    return output_path

# Function to add both methods to the ElectricityPriceModel class
def extend_model_with_fixed_interactions(ElectricityPriceModel):
    """Add fixed interaction methods to the ElectricityPriceModel class"""
    
    def calculate_interaction_values(self):
        """
        Calculate true SHAP interaction values.
        This is computationally expensive but necessary for proper interaction analysis.
        
        Returns:
            self (for method chaining)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first with explain_with_shap()")
            
        if not hasattr(self, 'explainer'):
            self.explainer = shap.TreeExplainer(self.model)
            
        print("Calculating SHAP interaction values (this may take some time)...")
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)
        print("SHAP interaction values calculated.")
        return self
    
    def plot_interaction_grid(self, output_dir="./plots/", filename="fixed_interaction_grid.png", dpi=300):
        """
        Create comprehensive SHAP dependency and interaction grid with TRUE interaction values.
        
        Arguments:
            output_dir: Directory to save the output
            filename: Filename for the saved plot
            dpi: Resolution for the saved image
            
        Returns:
            Path to the saved plot
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Calculate interaction values if not already done
        if not hasattr(self, 'shap_interaction_values'):
            self.calculate_interaction_values()
            
        return create_fixed_interaction_grid(self, output_path, dpi)
    
    # Add single interaction plot for individual features
    def plot_single_interaction(self, feature1, feature2, output_dir="./plots/", 
                               filename=None, figsize=(10, 8), dpi=300):
        """
        Plot a single SHAP interaction plot between two features.
        
        Arguments:
            feature1: First feature name
            feature2: Second feature name
            output_dir: Directory to save output
            filename: Filename (defaults to interaction_{feature1}_{feature2}.png)
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            Path to saved figure
        """
        if not hasattr(self, 'shap_interaction_values'):
            self.calculate_interaction_values()
            
        if filename is None:
            filename = f"interaction_{feature1}_{feature2}.png"
            
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Get feature indices
        feature_indices = {feature: list(self.X_test.columns).index(feature) 
                           for feature in self.X_test.columns}
                           
        idx1 = feature_indices[feature1]
        idx2 = feature_indices[feature2]
        
        # Plot using the standard SHAP method for interaction plots
        plt.figure(figsize=figsize)
        shap.dependence_plot(
            (idx1, idx2),  # Using tuple notation for interaction
            self.shap_interaction_values,
            self.X_test,
            display_features=self.X_test,
            show=False
        )
        
        # Customize title and axes
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"SHAP interaction value for\n{feature1} and {feature2}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP interaction plot between {feature1} and {feature2} saved to {output_path}")
        return output_path
    
    # Add the methods to the class
    ElectricityPriceModel.calculate_interaction_values = calculate_interaction_values
    ElectricityPriceModel.plot_interaction_grid = plot_interaction_grid
    ElectricityPriceModel.plot_single_interaction = plot_single_interaction
    
    return ElectricityPriceModel

# Example usage to create the plots
"""
from electricity_price_model import ElectricityPriceModel
from fixed_true_interactions import extend_model_with_fixed_interactions

# Extend the model
ElectricityPriceModel = extend_model_with_fixed_interactions(ElectricityPriceModel)

# Train model and calculate SHAP values
model = ElectricityPriceModel(random_state=42)
model.train(dataset, features=valid_features)
model.explain_with_shap()

# Calculate interaction values (this is computationally expensive)
model.calculate_interaction_values()

# Create the comprehensive grid plot with TRUE interaction values
model.plot_interaction_grid()

# Create a single interaction plot like Image 1
model.plot_single_interaction("load_forecast", "wind_forecast")
"""