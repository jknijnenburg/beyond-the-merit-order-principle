#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 5: SHAP Dependency Plots for Total Generation Ramp

This module creates SHAP dependency plots for the Total Generation Ramp feature,
showing the effect with and without interactions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import types

def create_generation_ramp_dependency_plots(model, output_path=None, dpi=300, 
                                           feature_name=None, figsize=(12, 5)):
    """
    Create SHAP dependency plots for the generation ramp feature,
    showing one plot with all interactions and one without.
    
    Arguments:
        model: Trained ElectricityPriceModel with SHAP values calculated
        output_path: Path to save the output figure
        dpi: Resolution for saved figure
        feature_name: Name of the feature to plot (will try to find a ramp feature if None)
        figsize: Figure size in inches
        
    Returns:
        Path to the saved figure
    """
    if model.shap_values is None:
        raise ValueError("SHAP values must be calculated first.")
    
    # Find a suitable ramp feature if not specified
    if feature_name is None:
        ramp_features = [col for col in model.X_test.columns if 'ramp' in col.lower()]
        generation_ramp_features = [col for col in ramp_features if 'generation' in col.lower()]
        
        if generation_ramp_features:
            feature_name = generation_ramp_features[0]
        elif ramp_features:
            feature_name = ramp_features[0]
        else:
            raise ValueError("No ramp features found in the dataset. Please specify a feature name.")
    
    # Check if the feature exists in the model
    if feature_name not in model.X_test.columns:
        available_ramp_features = [f for f in model.X_test.columns if 'ramp' in f]
        if available_ramp_features:
            print(f"Feature '{feature_name}' not found. Available ramp features: {available_ramp_features}")
            feature_name = available_ramp_features[0]
            print(f"Using '{feature_name}' instead.")
        else:
            raise ValueError(f"Feature '{feature_name}' not found and no alternative ramp features available.")
    
    # Get feature index
    feature_idx = list(model.X_test.columns).index(feature_name)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Display name with proper formatting
    if '_' in feature_name:
        words = feature_name.split('_')
        display_name = ' '.join(word.capitalize() for word in words)
    else:
        display_name = feature_name.capitalize()
    
    # Find a suitable interaction feature (e.g., load_forecast)
    interaction_feature = None
    for candidate in ['load_forecast', 'solar_forecast', 'wind_forecast']:
        if candidate in model.X_test.columns and candidate != feature_name:
            interaction_feature = candidate
            break
    
    # Define custom color map similar to the example
    colors = ["#1E3A8A", "#2563EB", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE", "#F9FAFB"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    # Plot 1: With all interactions
    interaction_idx = list(model.X_test.columns).index(interaction_feature) if interaction_feature else None
    
    # Extract feature data
    feature_values = model.X_test.iloc[:, feature_idx].values
    shap_values = model.shap_values[:, feature_idx]
    
    # Left plot with interactions
    plt.sca(axes[0])
    if interaction_idx is not None:
        # With interaction coloring
        interaction_values = model.X_test.iloc[:, interaction_idx].values
        axes[0].scatter(
            feature_values, 
            shap_values, 
            c=interaction_values, 
            cmap=custom_cmap,
            s=9,  # Smaller point size to match reference image
            alpha=0.8
        )
    else:
        # No suitable interaction feature found
        axes[0].scatter(
            feature_values, 
            shap_values, 
            color="#1E3A8A",
            s=9,
            alpha=0.8
        )
    
    # Customize plot 1
    axes[0].set_title("With all Interactions", fontsize=12)
    axes[0].set_xlabel(f"{display_name}\nday-ahead [MWh]", fontsize=10)
    axes[0].set_ylabel("SHAP values (Electricity Price)\n[EUR/MWh]", fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Without interactions
    plt.sca(axes[1])
    # Simple scatter plot without interaction coloring
    axes[1].scatter(
        feature_values, 
        shap_values, 
        color="#1E3A8A",
        s=9,
        alpha=0.8
    )
    
    # Customize plot 2
    axes[1].set_title("Without Interactions", fontsize=12)
    axes[1].set_xlabel(f"{display_name}\nday-ahead [MWh]", fontsize=10)
    axes[1].set_ylabel("", fontsize=10)  # No y-label on right plot
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Set consistent y-axis limits
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    # Set equal x-axis limits as well for better comparison
    x_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    x_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    axes[0].set_xlim(x_min, x_max)
    axes[1].set_xlim(x_min, x_max)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is None:
        os.makedirs("./plots/", exist_ok=True)
        output_path = "./plots/figure5_generation_ramp.png"
        
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Generation ramp dependency plots saved to {output_path}")
    return output_path


def create_detailed_dependency_plots(model, output_path=None, dpi=300, 
                                    feature_name=None,
                                    with_interaction_feature=None,
                                    use_shap_interaction_values=True):
    """
    Create more detailed SHAP dependency plots for the generation ramp feature,
    with explicit control over interactions.
    
    This version provides more control over the interaction representation,
    including the option to use true SHAP interaction values if available.
    
    Arguments:
        model: Trained ElectricityPriceModel with SHAP values calculated
        output_path: Path to save the output figure
        dpi: Resolution for saved figure
        feature_name: Name of the feature to plot
        with_interaction_feature: Name of feature to use for interaction in first plot
        use_shap_interaction_values: Whether to use true SHAP interaction values
            (requires model.shap_interaction_values to be calculated)
            
    Returns:
        Path to the saved figure
    """
    if model.shap_values is None:
        raise ValueError("SHAP values must be calculated first.")
    
    # Find a suitable ramp feature if not specified
    if feature_name is None:
        ramp_features = [col for col in model.X_test.columns if 'ramp' in col.lower()]
        generation_ramp_features = [col for col in ramp_features if 'generation' in col.lower()]
        
        if generation_ramp_features:
            feature_name = generation_ramp_features[0]
        elif ramp_features:
            feature_name = ramp_features[0]
        else:
            raise ValueError("No ramp features found in the dataset. Please specify a feature name.")
        
    # Check if the feature exists in the model
    if feature_name not in model.X_test.columns:
        available_ramp_features = [f for f in model.X_test.columns if 'ramp' in f]
        if available_ramp_features:
            print(f"Feature '{feature_name}' not found. Available ramp features: {available_ramp_features}")
            feature_name = available_ramp_features[0]
            print(f"Using '{feature_name}' instead.")
        else:
            raise ValueError(f"Feature '{feature_name}' not found and no alternative ramp features available.")
    
    # Get feature index
    feature_idx = list(model.X_test.columns).index(feature_name)
    
    # Check interaction feature
    if with_interaction_feature is not None and with_interaction_feature not in model.X_test.columns:
        print(f"Interaction feature '{with_interaction_feature}' not found.")
        with_interaction_feature = None
    
    # Find a suitable interaction feature if not specified
    if with_interaction_feature is None:
        for candidate in ['load_forecast', 'solar_forecast', 'wind_forecast']:
            if candidate in model.X_test.columns and candidate != feature_name:
                with_interaction_feature = candidate
                break
    
    # Get interaction feature index if available
    interaction_idx = None
    if with_interaction_feature:
        interaction_idx = list(model.X_test.columns).index(with_interaction_feature)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display name with proper formatting
    if '_' in feature_name:
        words = feature_name.split('_')
        display_name = ' '.join(word.capitalize() for word in words)
    else:
        display_name = feature_name.capitalize()
    
    # Define custom color map similar to the example
    colors = ["#1E3A8A", "#2563EB", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE", "#F9FAFB"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
    
    # Calculate interaction values if needed and not already done
    if use_shap_interaction_values and not hasattr(model, 'shap_interaction_values'):
        try:
            if hasattr(model, 'calculate_interaction_values'):
                model.calculate_interaction_values()
            else:
                # Try to inject the method for calculating interaction values
                print("Calculating SHAP interaction values (this may take some time)...")
                if not hasattr(model, 'explainer'):
                    model.explainer = shap.TreeExplainer(model.model)
                model.shap_interaction_values = model.explainer.shap_interaction_values(model.X_test)
                print("SHAP interaction values calculated.")
        except Exception as e:
            print(f"Could not calculate SHAP interaction values: {str(e)}")
            use_shap_interaction_values = False
    
    # Plot 1: With all interactions
    plt.sca(axes[0])
    
    # Extract feature data
    feature_values = model.X_test.iloc[:, feature_idx].values
    
    if use_shap_interaction_values and hasattr(model, 'shap_interaction_values') and model.shap_interaction_values is not None and interaction_idx is not None:
        # We have interaction values, we can create a more accurate plot
        
        # For interaction plot, use the main SHAP value (diagonal of interaction matrix)
        # plus the interaction with the specified feature
        main_effect = model.shap_interaction_values[:, feature_idx, feature_idx]
        
        if interaction_idx is not None:
            # Add the interaction effect (both directions as interaction matrix is symmetric)
            interaction_effect = (
                model.shap_interaction_values[:, feature_idx, interaction_idx] + 
                model.shap_interaction_values[:, interaction_idx, feature_idx]
            )
            total_effect = main_effect + interaction_effect
        else:
            total_effect = main_effect
            
        # Coloring by the interaction feature
        if interaction_idx is not None:
            color_values = model.X_test.iloc[:, interaction_idx].values
            scatter = axes[0].scatter(
                feature_values,
                total_effect,
                c=color_values,
                cmap=custom_cmap,
                s=9,
                alpha=0.8
            )
        else:
            axes[0].scatter(
                feature_values,
                total_effect,
                color="#1E3A8A",
                s=9,
                alpha=0.8
            )
    else:
        # Fallback to standard SHAP values
        shap_values = model.shap_values[:, feature_idx]
        
        if interaction_idx is not None:
            # With interaction coloring
            interaction_values = model.X_test.iloc[:, interaction_idx].values
            axes[0].scatter(
                feature_values, 
                shap_values, 
                c=interaction_values, 
                cmap=custom_cmap,
                s=9,
                alpha=0.8
            )
        else:
            # No interaction coloring
            axes[0].scatter(
                feature_values, 
                shap_values,
                color="#1E3A8A",
                s=9,
                alpha=0.8
            )
    
    # Customize plot 1
    axes[0].set_title("With all Interactions", fontsize=12)
    axes[0].set_xlabel(f"{display_name}\nday-ahead [MWh]", fontsize=10)
    axes[0].set_ylabel("SHAP values (Electricity Price)\n[EUR/MWh]", fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Without interactions
    plt.sca(axes[1])
    
    if use_shap_interaction_values and hasattr(model, 'shap_interaction_values') and model.shap_interaction_values is not None:
        # Just use the main effect (diagonal of interaction matrix)
        main_effect = model.shap_interaction_values[:, feature_idx, feature_idx]
        
        axes[1].scatter(
            feature_values,
            main_effect,
            color="#1E3A8A",
            s=9,
            alpha=0.8
        )
    else:
        # Fallback to standard SHAP values
        shap_values = model.shap_values[:, feature_idx]
        
        axes[1].scatter(
            feature_values, 
            shap_values,
            color="#1E3A8A",
            s=9,
            alpha=0.8
        )
    
    # Customize plot 2
    axes[1].set_title("Without Interactions", fontsize=12)
    axes[1].set_xlabel(f"{display_name}\nday-ahead [MWh]", fontsize=10)
    axes[1].set_ylabel("", fontsize=10)  # No y-label on right plot
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Set consistent y-axis limits
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    # Set equal x-axis limits as well for better comparison
    x_min = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    x_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    axes[0].set_xlim(x_min, x_max)
    axes[1].set_xlim(x_min, x_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if output_path is None:
        os.makedirs("./plots/", exist_ok=True)
        output_path = "./plots/figure5_detailed_dependency.png"
        
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed dependency plots saved to {output_path}")
    return output_path


# Integration with the unified visualization system
def create_unified_visualizations(model_input, dataset=None, plot_dir="./plots/", 
                                slopes_data=None, prefix="fig5_"):
    """
    Create Figure 5 visualizations compatible with the unified interface.
    
    Args:
        model_input: Single model or list of models
        dataset: Dataset (required if model_input is a list)
        plot_dir: Directory to save plots
        slopes_data: Not used for this figure
        prefix: Prefix for saved files
        
    Returns:
        Path to the main visualization
    """
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
            
        print(f"Creating Figure 5 visualizations using ensemble approach with {len(all_models)} models...")
        
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
        print("Creating Figure 5 visualizations using single model approach...")
        model_obj = model_input
    
    # Ensure SHAP values are calculated
    if not hasattr(model_obj, 'shap_values') or model_obj.shap_values is None:
        model_obj.explain_with_shap()
    
    # Create simple version first
    simple_output_path = os.path.join(plot_dir, f"{prefix}generation_ramp.png")
    create_generation_ramp_dependency_plots(
        model_obj,
        output_path=simple_output_path
    )
    
    # Create detailed version if interaction values are available or can be calculated
    try:
        detailed_output_path = os.path.join(plot_dir, f"{prefix}detailed_dependency.png")
        create_detailed_dependency_plots(
            model_obj,
            output_path=detailed_output_path
        )
        return detailed_output_path
    except Exception as e:
        print(f"Could not create detailed dependency plots: {str(e)}")
        return simple_output_path


if __name__ == "__main__":
    print("Figure 5 module - run with a model to create visualizations")
    print("Example usage:")
    print("  from electricity_price_model import ElectricityPriceModel")
    print("  model = ElectricityPriceModel()")
    print("  model.explain_with_shap()")
    print("  create_generation_ramp_dependency_plots(model)")