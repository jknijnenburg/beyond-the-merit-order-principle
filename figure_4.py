#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified visualization module for electricity price models.
Handles both single models and ensembles of consistent models.

This module provides comprehensive SHAP dependency and interaction visualizations
with support for both individual models and ensemble approaches.
"""

import os
import types
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Import figure_3 for dependency plots
try:
    import figure_3
except ImportError:
    print("Warning: figure_3 module not found. Some visualizations may not be available.")


### INTERACTION GRID VISUALIZATION ###
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
    
    # Filter to only available features
    available_main_features = [f for f in main_features if f in model.X_test.columns]
    
    if len(available_main_features) == 0:
        print("Warning: None of the main features are available in the dataset.")
        return None
        
    # Update interaction features based on available columns
    for feature in interaction_features:
        interaction_features[feature] = [f for f in interaction_features[feature] 
                                       if f in model.X_test.columns]
    
    # Display names for features
    feature_display_names = {
        "load_forecast": "Load day-ahead [MWh]",
        "solar_forecast": "Solar day-ahead [MWh]",
        "wind_forecast": "Wind day-ahead [MWh]",
        "net_import_export": "Import export day-ahead [MWh]",
        "oil": "Oil price",
        "natural_gas": "Natural gas price"
    }
    
    # Create figure with subplots - only for available features
    n_rows = len(available_main_features)
    fig, axs = plt.subplots(nrows=n_rows, ncols=5, figsize=(18, 3.5*n_rows))
    
    # Handle the case of a single row
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    # Row labels
    row_labels = ['a', 'b', 'c', 'd']
    
    # Add column headers (in German)
    fig.text(0.17, 0.98, "Mit allen Interaktionen", ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.38, 0.98, "Ohne Interaktionen", ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.7, 0.98, "Interaktionen mit drei ausgewählten anderen Features", ha='center', va='center', fontsize=12, fontweight='bold')

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
            main_effect = model.shap_interaction_values[:, feature_idx, feature_idx]
            # scatter = ax.scatter(feature_values, shap_values, 
            #                     color="#1F77B4", s=5, alpha=0.8)
            scatter = ax.scatter(feature_values, main_effect, 
                                 color="#1F77B4", s=5, alpha=0.8)
        
        ax.set_xlabel(feature_display_names.get(model.X_test.columns[feature_idx], 
                                            model.X_test.columns[feature_idx]))
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
        
        ax.set_xlabel(feature_display_names.get(model.X_test.columns[main_idx], 
                                            model.X_test.columns[main_idx]))
        ax.grid(True, linestyle='--', alpha=0.3)
        return scatter
    
    # Set y-axis labels for the first column
    y_labels_first_col = [
        "SHAP values (Electricity Price)\n[EUR/MWh]", 
        "SHAP values (Electricity Price)\n[EUR/MWh]",
        "SHAP values (Electricity Price)\n[EUR/MWh]",
        "SHAP values (Electricity Price)\n[EUR/MWh]"
    ]
    
    # Process each main feature (rows)
    for row, main_feature in enumerate(available_main_features):
        main_idx = feature_indices[main_feature]
        
        # Add row label
        axs[row, 0].text(-0.3, 0.5, row_labels[row], transform=axs[row, 0].transAxes,
                fontsize=12, fontweight='bold', va='center')
                
        # Column 1: Plot with all interactions (using strongest interaction for coloring)
        available_interactions = interaction_features[main_feature]
        if available_interactions:
            strongest_interaction = available_interactions[0]
            interact_idx = feature_indices[strongest_interaction]
            create_regular_plot(axs[row, 0], main_idx, interact_idx)
        else:
            # No interactions available, use regular plot without coloring
            create_regular_plot(axs[row, 0], main_idx, None)
        
        # Set y-label only for the first column
        axs[row, 0].set_ylabel(y_labels_first_col[min(row, len(y_labels_first_col)-1)])
        
        # Column 2: Plot without interactions
        create_regular_plot(axs[row, 1], main_idx, None)
        
        # Remove y-label for column 2
        axs[row, 1].set_ylabel("")
        
        # Columns 3-5: TRUE interaction plots with specific features
        for col, interaction_feature in enumerate(interaction_features[main_feature][:3]):  # Up to 3 interactions
            if interaction_feature not in model.X_test.columns:
                # Skip if interaction feature is not available
                axs[row, col+2].set_visible(False)
                continue
                
            interact_idx = feature_indices[interaction_feature]
            col_idx = col + 2  # Adjust for 0-based indexing
            
            # Add title for each cell based on the interaction feature
            axs[row, col_idx].set_title(
                feature_display_names.get(interaction_feature, interaction_feature), 
                fontsize=10
            )
            
            # Create the TRUE interaction plot
            create_interaction_plot(axs[row, col_idx], main_idx, interact_idx)
            
            # Remove y-label for columns 3-5
            axs[row, col_idx].set_ylabel("")
    
    # Add a colorbar for the entire figure
    # Create a new axis for the colorbar that doesn't overlap with the subplots
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjusted position
    
    # Create the colorbar with the correct normalization
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the colormap
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Wert des interagierenden Features')  # German label
    
    # Add "Niedrig-Hoch" labels to the colorbar (German)
    cbar_ax.text(0.5, 0.02, 'Niedrig', ha='center', va='bottom', transform=cbar_ax.transAxes)
    cbar_ax.text(0.5, 0.98, 'Hoch', ha='center', va='top', transform=cbar_ax.transAxes)
    
    # Adjust layout - increase right margin to make room for colorbar
    plt.subplots_adjust(left=0.08, right=0.91, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed SHAP interaction grid saved to {output_path}")
    return output_path


### UNIFIED VISUALIZATION SYSTEM ###
def create_unified_visualizations(model_input, dataset=None, plot_dir="./plots/"):
    """
    Unified visualization function that works with both single models and model ensembles.
    
    Args:
        model_input: Either a single ElectricityPriceModel or a list of model dictionaries
                   from the consistency analysis
        dataset: The dataset used for testing, required if model_input is a list of models
        plot_dir: Directory where plots will be saved
        
    Returns:
        Path to the main interaction grid visualization
    """
    print("\n=== SCHRITT 7: UNIFIED VISUALIZATIONS ===\n")
    
    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)
    
    # Determine if we're dealing with a single model or an ensemble
    is_ensemble = isinstance(model_input, list)
    
    if is_ensemble:
        # Handle ensemble of models
        all_models = model_input
        
        if not all_models or len(all_models) == 0:
            print("No models available for visualization!")
            return None
            
        print(f"Creating visualizations using ensemble approach with {len(all_models)} models...")
        
        # Select the best model from the ensemble
        best_model_info = sorted(all_models, key=lambda x: x['score'])[0]
        best_model = best_model_info['model']
        print(f"Best model selected (Score: {best_model_info['score']:.4f})")
        
        # Create an ElectricityPriceModel object with this model
        from electricity_price_model import ElectricityPriceModel
        model_obj = ElectricityPriceModel(random_state=42)
        model_obj.model = best_model
        
        # Set test data
        if dataset is None:
            raise ValueError("Dataset is required when using an ensemble of models")
            
        features = list(best_model_info['shap_importance'].keys())
        model_obj.X_test = dataset[features].copy()
        model_obj.y_test = dataset['price'] if 'price' in dataset.columns else None
        
    else:
        # Handle single model
        print("Creating visualizations using single model approach...")
        model_obj = model_input
        
    # Ensure SHAP values are calculated
    print("Calculating SHAP values for visualizations...")
    if not hasattr(model_obj, 'shap_values') or model_obj.shap_values is None:
        model_obj.explain_with_shap()
    
    # 1. Feature Importance Plot
    print("Creating feature importance plot...")
    importance_df = model_obj.plot_global_feature_importance()
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
    plt.close()
    print("Feature importance plot saved")
    
    # Print feature importance ranking
    print("\nFeature Importance Ranking:")
    for _, row in importance_df.iterrows():
        print(f" {row['Rank']}. {row['Feature']}: {row['Importance']:.3f}")
    
    # 2. SHAP Dependency Plots
    try:
        print("\nCreating SHAP dependency plots...")
        if 'figure_3' in globals() or 'figure_3' in locals():
            top_features = figure_3.plot_paper_style_shap_dependencies(model_obj)
        else:
            # Fallback using model's built-in method if available
            top_features = model_obj.plot_paper_style_shap_dependencies(top_n=3)
        
        plt.savefig(os.path.join(plot_dir, "shap_dependency.png"), dpi=300)
        plt.close()
        print("SHAP dependency plots saved")
    except Exception as e:
        print(f"Error creating SHAP dependency plots: {str(e)}")
        top_features = []
    
    # 3. Calculate SHAP interaction values (needed for interaction plots)
    print("\nCalculating SHAP interaction values (this may take some time)...")
    try:
        # Add the calculate_interaction_values method if it doesn't exist
        if not hasattr(model_obj, 'calculate_interaction_values'):
            model_obj.calculate_interaction_values = types.MethodType(
                lambda self: self._calculate_interaction_values() if hasattr(self, '_calculate_interaction_values')
                else self._fallback_calculate_interaction_values(),
                model_obj
            )
            
            # Add fallback method
            model_obj._fallback_calculate_interaction_values = types.MethodType(
                lambda self: self._calculate_interaction_values_internal(),
                model_obj
            )
            
            # Internal calculation method
            model_obj._calculate_interaction_values_internal = types.MethodType(
                lambda self: self._init_and_calculate_interactions(),
                model_obj
            )
            
            # Initialize explainer and calculate values
            model_obj._init_and_calculate_interactions = types.MethodType(
                lambda self: self._set_interaction_values(),
                model_obj
            )
            
            # Set interaction values
            model_obj._set_interaction_values = types.MethodType(
                lambda self: self._finalize_interaction_calculation(),
                model_obj
            )
            
            # Final calculation step
            model_obj._finalize_interaction_calculation = types.MethodType(
                lambda self: setattr(self, 'shap_interaction_values', 
                                    (self.explainer.shap_interaction_values(self.X_test) 
                                    if hasattr(self, 'explainer') 
                                    else shap.TreeExplainer(self.model).shap_interaction_values(self.X_test))),
                model_obj
            )
        
        # Calculate interaction values
        if not hasattr(model_obj, 'shap_interaction_values') or model_obj.shap_interaction_values is None:
            if not hasattr(model_obj, 'explainer'):
                model_obj.explainer = shap.TreeExplainer(model_obj.model)
            model_obj.shap_interaction_values = model_obj.explainer.shap_interaction_values(model_obj.X_test)
        
        print("SHAP interaction values calculated.")
    except Exception as e:
        print(f"Error calculating SHAP interaction values: {str(e)}")
        traceback.print_exc()
    
    # 4. Create comprehensive interaction grid
    print("\nCreating SHAP interaction grid...")
    grid_output_path = os.path.join(plot_dir, "interaction_grid.png")
    
    try:
        # Add plot_interaction_grid method if it doesn't exist
        if not hasattr(model_obj, 'plot_interaction_grid'):
            model_obj.plot_interaction_grid = types.MethodType(
                lambda self, output_dir="./plots/", filename="interaction_grid.png", dpi=300: 
                create_fixed_interaction_grid(self, 
                                           output_path=os.path.join(output_dir, filename), 
                                           dpi=dpi),
                model_obj
            )
        
        # Create the interaction grid
        interaction_grid_path = model_obj.plot_interaction_grid(
            output_dir=plot_dir,
            filename="interaction_grid.png",
            dpi=300
        )
        print(f"SHAP interaction grid saved to {interaction_grid_path}")
    except Exception as e:
        print(f"Error creating interaction grid: {str(e)}")
        traceback.print_exc()
        interaction_grid_path = None
    
    # 5. Create individual interaction plots for key feature pairs
    print("\nCreating individual interaction plots...")
    
    # Add single interaction plot method if it doesn't exist
    if not hasattr(model_obj, 'plot_single_interaction'):
        model_obj.plot_single_interaction = types.MethodType(
            lambda self, feature1, feature2, output_dir=plot_dir, 
                    filename=None, figsize=(10, 8), dpi=300:
            _create_single_interaction_plot(self, feature1, feature2, 
                                         output_dir, filename, figsize, dpi),
            model_obj
        )
    
    # Key feature pairs to visualize
    key_features = ["load_forecast", "solar_forecast", "wind_forecast", "net_import_export"]
    created_interactions = []
    
    # Create interaction plots for available feature pairs
    for i, feature1 in enumerate(key_features):
        if feature1 not in model_obj.X_test.columns:
            continue
        for feature2 in key_features[i+1:]:
            if feature2 not in model_obj.X_test.columns:
                continue
                
            try:
                output_path = model_obj.plot_single_interaction(
                    feature1=feature1, 
                    feature2=feature2
                )
                created_interactions.append(output_path)
                print(f"Interaction plot created: {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Error creating interaction plot for {feature1}-{feature2}: {str(e)}")
    
    # 6. Create performance plot (actual vs predicted)
    try:
        y_pred = model_obj.predict(model_obj.X_test)
        y_true = model_obj.y_test
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
        plt.xlabel("Tatsächlicher Preis (EUR/MWh)")
        plt.ylabel("Vorhergesagter Preis (EUR/MWh)")
        plt.title("Tatsächliche vs. vorhergesagte Elektrizitätspreise")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        performance_path = os.path.join(plot_dir, "actual_vs_predicted.png")
        plt.savefig(performance_path)
        plt.close()
        print(f"Performance plot saved to {performance_path}")
    except Exception as e:
        print(f"Error creating performance plot: {str(e)}")
    
    # Return the path to the interaction grid
    return interaction_grid_path


# Helper function for creating individual interaction plots
def _create_single_interaction_plot(model, feature1, feature2, output_dir="./plots", 
                                 filename=None, figsize=(10, 8), dpi=300):
    """
    Create a single SHAP interaction plot between two features.
    
    Args:
        model: Model with shap_interaction_values calculated
        feature1: First feature name
        feature2: Second feature name
        output_dir: Directory to save the plot
        filename: Optional custom filename
        figsize: Figure size
        dpi: Resolution
        
    Returns:
        Path to the saved plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"interaction_{feature1}_{feature2}.png"
    
    # Full output path
    output_path = os.path.join(output_dir, filename)
    
    # Get feature indices
    feature_indices = {feature: list(model.X_test.columns).index(feature) 
                      for feature in model.X_test.columns}
                      
    # Check if features exist
    if feature1 not in feature_indices or feature2 not in feature_indices:
        raise ValueError(f"Features not found in dataset: {feature1}, {feature2}")
        
    idx1 = feature_indices[feature1]
    idx2 = feature_indices[feature2]
    
    # Calculate interaction values if not already done
    if not hasattr(model, 'shap_interaction_values') or model.shap_interaction_values is None:
        if not hasattr(model, 'explainer'):
            model.explainer = shap.TreeExplainer(model.model)
        model.shap_interaction_values = model.explainer.shap_interaction_values(model.X_test)
    
    # Create interaction plot
    plt.figure(figsize=figsize)
    shap.dependence_plot(
        (idx1, idx2),  # Tuple notation for interaction
        model.shap_interaction_values,
        model.X_test,
        display_features=model.X_test,
        show=False
    )
    
    # Customize titles and labels
    plt.xlabel(f"{feature1}")
    plt.ylabel(f"SHAP-Interaktionswert für\n{feature1} und {feature2}")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


# Legacy function names for backward compatibility
def extend_model_with_fixed_interactions(ElectricityPriceModel):
    """
    Legacy function to add fixed interaction methods to the ElectricityPriceModel class.
    Maintained for backward compatibility.
    """
    def calculate_interaction_values(self):
        """Calculate true SHAP interaction values"""
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first with explain_with_shap()")
            
        if not hasattr(self, 'explainer'):
            self.explainer = shap.TreeExplainer(self.model)
            
        print("Calculating SHAP interaction values (this may take some time)...")
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)
        print("SHAP interaction values calculated.")
        return self
    
    def plot_interaction_grid(self, output_dir="./plots/", filename="fixed_interaction_grid.png", dpi=300):
        """Create comprehensive SHAP dependency and interaction grid"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Calculate interaction values if not already done
        if not hasattr(self, 'shap_interaction_values'):
            self.calculate_interaction_values()
            
        return create_fixed_interaction_grid(self, output_path, dpi)
    
    def plot_single_interaction(self, feature1, feature2, output_dir="./plots/", 
                              filename=None, figsize=(10, 8), dpi=300):
        """Plot a single SHAP interaction plot between two features"""
        return _create_single_interaction_plot(
            self, feature1, feature2, output_dir, filename, figsize, dpi
        )
    
    # Add methods to the class
    ElectricityPriceModel.calculate_interaction_values = calculate_interaction_values
    ElectricityPriceModel.plot_interaction_grid = plot_interaction_grid
    ElectricityPriceModel.plot_single_interaction = plot_single_interaction
    
    return ElectricityPriceModel


# For backward compatibility
def create_consistent_visualizations(all_models, dataset, plot_dir="./plots/"):
    """
    Legacy function for creating visualizations using consistent models.
    Forwards to the unified visualization function.
    """
    return create_unified_visualizations(all_models, dataset, plot_dir)