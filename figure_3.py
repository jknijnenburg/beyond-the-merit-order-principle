#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 3: SHAP Dependency Plots and Slope Violin Plots

This module creates the combined visualization of:
1. SHAP dependency plots for load, solar, and wind forecasts (top row)
2. Violin plot showing the distribution of dependency slopes (bottom row)

The calculations for the slopes can be performed separately and passed to these
visualization functions, allowing for flexible use in both single-model and
multi-model scenarios.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress


### FIGURE 3 DEPENDENCY PLOTS ###
def plot_paper_style_shap_dependencies(model, top_n=3, ax=None, fig=None):
    """
    Plot SHAP dependency plots for top features in a row,
    matching the style from the reference paper.

    Arguments:
        model: Trained ElectricityPriceModel with SHAP values calculated
        top_n: Number of top features to plot
        ax: Optional axes array to plot on (for combined plots)
        fig: Optional figure to use (for combined plots)

    Returns:
        List of top feature names and slope values dictionary
    """
    if model.shap_values is None:
        raise ValueError("SHAP values must be calculated first.")

    # Use specific features instead of top features by importance
    top_features = ["load_forecast", "solar_forecast", "wind_forecast"]

    # Filter to only available features
    available_features = [f for f in top_features if f in model.X_test.columns]

    if not available_features:
        # Fall back to top features by importance if requested ones aren't available
        feature_importance = np.abs(model.shap_values).mean(0)
        top_indices = np.argsort(-feature_importance)[:top_n]
        available_features = [model.X_test.columns[i] for i in top_indices]
        print(
            f"None of the requested features available. Using top features by importance: {available_features}"
        )

    print(f"Creating SHAP dependency plots for: {available_features}")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, len(available_features), figsize=(18, 6))
        # Convert to array for consistent indexing if only one plot
        if len(available_features) == 1:
            ax = np.array([ax])

    # Define prettier feature names for x-axis labels
    feature_labels = {
        "load_forecast": "Load day-ahead [MWh]",
        "wind_forecast": "Wind day-ahead [MWh]",
        "solar_forecast": "Solar day-ahead [MWh]",
    }

    # Track slope values for each feature
    slopes = {}

    # Create plots
    for i, feature in enumerate(available_features):
        feature_idx = list(model.X_test.columns).index(feature)

        # Make a copy of the data for potential negation
        x_data = model.X_test[feature].copy()

        # Negate solar and wind values to match paper style
        if feature in ["solar_forecast", "wind_forecast"]:
            x_data = -x_data

        # Force matplotlib to use our specific axis
        plt.sca(ax[i])

        # Plot SHAP values against the feature data
        scatter = ax[i].scatter(
            x_data,
            model.shap_values[:, feature_idx],
            alpha=0.6,
            s=12,  # Smaller point size for cleaner look
            color="#333333",  # Darker points like in the paper
        )

        # Add a trend line
        slope, intercept, r_value, p_value, std_err = linregress(
            x_data, model.shap_values[:, feature_idx]
        )
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        ax[i].plot(x_range, intercept + slope * x_range, color="skyblue", linewidth=2)

        # Update plot style to match paper
        ax[i].set_title(f"{chr(97+i)}", loc="left", fontsize=14, fontweight="bold")

        # Use prettier x-axis label if available
        ax[i].set_xlabel(feature_labels.get(feature, feature), fontsize=12)

        # Only add y-axis label to the first plot
        if i == 0:
            ax[i].set_ylabel("SHAP values (Electricity Price)\n[EUR/MWh]", fontsize=12)

        # Add grid for better readability
        ax[i].grid(True, alpha=0.3, linestyle="--")

        # Store the slope value
        # If we negated the x values, we need to negate the slope too for consistency
        if feature in ["solar_forecast", "wind_forecast"]:
            slopes[feature] = -slope
        else:
            slopes[feature] = slope

    # Return top features and the slopes dictionary
    return available_features, slopes


### FIGURE 3 VIOLIN PLOTS ###
def create_slope_violin_plot(slopes_data, ax=None, feature_order=None):
    """
    Create a violin plot of SHAP dependency slopes matching the paper style.

    Arguments:
        slopes_data: Dictionary mapping feature names to arrays of slope values,
                    or a single model's slopes dictionary
        ax: Optional matplotlib axis to plot on
        feature_order: Optional list specifying the order of features

    Returns:
        The matplotlib axis with the plot
    """
    # Convert single slopes dict to the expected format if needed
    if all(isinstance(val, (int, float)) for val in slopes_data.values()):
        # This is a single model's slopes dict, convert to arrays with one value each
        slopes_collection = {k: np.array([v]) for k, v in slopes_data.items()}
    else:
        # Already in the right format
        slopes_collection = slopes_data

    # Map internal feature names to display names
    feature_display_names = {
        "load_forecast": "Load day-ahead",
        "solar_forecast": "Solar day-ahead",
        "wind_forecast": "Wind day-ahead",
    }

    # Use specified feature order or default
    if feature_order is None:
        filtered_features = ["load_forecast", "solar_forecast", "wind_forecast"]
    else:
        filtered_features = feature_order

    # Keep only features that exist in the slopes_collection
    feature_names = [name for name in filtered_features if name in slopes_collection]

    # If none of the requested features are available, print a warning
    if not feature_names:
        print("Warning: None of the specified features found in slopes data!")
        # Fall back to all features
        feature_names = list(slopes_collection.keys())

    # Get values and possibly negate renewable generation slopes as in the paper
    feature_values = []
    for name in feature_names:
        values = np.array(slopes_collection[name])
        # Check if renewable feature (solar or wind) AND if values are primarily negative
        if name in ["solar_forecast", "wind_forecast"] and np.median(values) < 0:
            values = -values  # Negate values for renewable generation as in the paper
            print(f"Negating values for {name} as per paper methodology")
        feature_values.append(values)

    positions = list(range(1, len(feature_names) + 1))

    # Create a new axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Set colors to match the reference image
    colors = ["#D5F0F2", "#65B1C1", "#3D629B"]  # Light blue, medium blue, dark blue

    # Create violin plot
    violin_parts = ax.violinplot(
        feature_values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Customize violin appearance
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    # Add box plots inside the violins - styled to match the paper
    boxplots = ax.boxplot(
        feature_values,
        positions=positions,
        widths=0.15,
        patch_artist=False,
        boxprops=dict(linestyle="-", linewidth=1.5, color="black"),
        whiskerprops=dict(linestyle="-", linewidth=1.5, color="black"),
        medianprops=dict(linestyle="-", linewidth=1.5, color="black"),
        capprops=dict(linestyle="-", linewidth=1.5, color="black"),
        flierprops=dict(marker=".", markerfacecolor="black", markersize=3),
    )

    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [feature_display_names.get(name, name) for name in feature_names], fontsize=12
    )
    ax.set_ylabel("Slope [EUR MWh$^{-2}$]", fontsize=12)  # Using LaTeX for superscript

    # Add the "d" label in top-left corner as in the paper
    ax.set_title("d", loc="left", fontweight="bold", fontsize=14)

    # Add grid lines (light gray, dashed)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Set y-axis limits to match data range with a small margin
    y_min = min([np.min(vals) for vals in feature_values])
    y_max = max([np.max(vals) for vals in feature_values])
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    # Make the plot background white and lighten the border
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")

    return ax


### COMBINED FIGURE 3 PLOT ###
def create_combined_dependency_violin_plot(
    model=None, slopes_data=None, output_path=None, dpi=300
):
    """
    Create the complete Figure 3 with SHAP dependency plots (top) and
    slope violin plot (bottom).

    Arguments:
        model: The trained model for dependency plots (required if no slopes_data)
        slopes_data: Pre-calculated slope data for the violin plot.
                    If None, will use slopes from a single model.
        output_path: Path to save the figure
        dpi: Resolution for saved figure

    Returns:
        Path to the saved figure
    """
    if model is None and slopes_data is None:
        raise ValueError("Either model or slopes_data must be provided")

    # If no output path provided, use default
    if output_path is None:
        # Create plots directory if it doesn't exist
        os.makedirs("./plots/", exist_ok=True)
        output_path = "./plots/figure3_combined.png"

    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=(18, 12))

    # Create gridspec with equal 50/50 split between dependency plots and violin plot
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])  # Equal height for both rows

    # Top row for dependency plots
    dep_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]

    # Bottom row for violin plot (spans all columns)
    violin_ax = fig.add_subplot(gs[1, :])

    # Create dependency plots
    if model is not None:
        # Ensure SHAP values are calculated
        if not hasattr(model, "shap_values") or model.shap_values is None:
            if hasattr(model, "explain_with_shap"):
                print("Calculating SHAP values...")
                model.explain_with_shap()
            else:
                raise ValueError(
                    "Model must have SHAP values calculated or explain_with_shap method"
                )

        # Create dependency plots
        features, model_slopes = plot_paper_style_shap_dependencies(
            model, top_n=3, ax=dep_axes, fig=fig
        )

        # If no slopes_data provided, use the slopes from this model
        if slopes_data is None:
            # Convert single model slopes to the format expected by create_slope_violin_plot
            slopes_data = {k: np.array([v]) for k, v in model_slopes.items()}

    # Create violin plot if we have slopes data
    if slopes_data is not None:
        create_slope_violin_plot(slopes_data, ax=violin_ax)

    # Add more bottom margin for the violin plot labels
    plt.subplots_adjust(bottom=0.15)
    
    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Combined Figure 3 saved to {output_path}")
    return output_path


### HELPER FUNCTION FOR SLOPE CALCULATIONS ###
def calculate_shap_dependency_slopes(model, required_features=None):
    """
    Calculate slopes of linear fits from SHAP dependency plots for specific features.

    Arguments:
        model: Trained ElectricityPriceModel with SHAP values calculated
        required_features: List of feature names to analyze (defaults to load_forecast, solar_forecast, wind_forecast)

    Returns:
        Dictionary mapping feature names to slope values
    """
    if model.shap_values is None:
        raise ValueError(
            "SHAP values must be calculated first with explain_with_shap()"
        )

    # Use specified features or default to these three key features
    if required_features is None:
        required_features = ["load_forecast", "solar_forecast", "wind_forecast"]

    available_features = [f for f in required_features if f in model.X_test.columns]

    if not available_features:
        print(
            f"Warning: None of the required features {required_features} are available in the model!"
        )
        # Fall back to top features by importance
        feature_importance = np.abs(model.shap_values).mean(0)
        top_indices = np.argsort(-feature_importance)[:3]
        available_features = [model.X_test.columns[i] for i in top_indices]
        print(f"Using top features by importance instead: {available_features}")

    # Calculate slopes for each feature
    slopes = {}

    for feature in available_features:
        try:
            feature_idx = list(model.X_test.columns).index(feature)

            # Get feature values and corresponding SHAP values
            x_values = model.X_test[feature].values
            y_values = model.shap_values[:, feature_idx]

            # Check if feature is solar or wind - apply negation as in paper if needed
            if feature in ["solar_forecast", "wind_forecast"]:
                x_values = -x_values  # Negate for consistency with paper

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

            # Store the slope - we applied negation to x, so the slope is already correctly oriented
            slopes[feature] = slope

        except Exception as e:
            print(f"Error calculating slope for {feature}: {str(e)}")

    return slopes


### Backward compatibility with unified visualization system ###
def create_unified_visualizations(
    model_input, dataset=None, plot_dir="./plots/", slopes_data=None, prefix="fig3_"
):
    """
    Create Figure 3 visualizations compatible with the unified interface.

    Args:
        model_input: Single model or list of models
        dataset: Dataset (required if model_input is a list)
        plot_dir: Directory to save plots
        slopes_data: Optional pre-calculated slopes data
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

        print(
            f"Creating Figure 3 visualizations using ensemble approach with {len(all_models)} models..."
        )

        # Select the best model from the ensemble
        best_model_info = sorted(all_models, key=lambda x: x["score"])[0]
        best_model = best_model_info["model"]
        print(f"Best model selected (Score: {best_model_info['score']:.4f})")

        # Create an ElectricityPriceModel object with this model
        from electricity_price_model import ElectricityPriceModel

        model_obj = ElectricityPriceModel(random_state=42)
        model_obj.model = best_model

        # Set test data
        if dataset is None:
            raise ValueError("Dataset is required when using an ensemble of models")

        features = list(best_model_info["shap_importance"].keys())
        model_obj.X_test = dataset[features].copy()
        model_obj.y_test = dataset["price"] if "price" in dataset.columns else None

    else:
        # Handle single model
        print("Creating Figure 3 visualizations using single model approach...")
        model_obj = model_input

    # Ensure SHAP values are calculated
    if not hasattr(model_obj, "shap_values") or model_obj.shap_values is None:
        model_obj.explain_with_shap()

    # If no slopes data provided, calculate from the single model
    if slopes_data is None and not is_ensemble:
        model_slopes = calculate_shap_dependency_slopes(model_obj)
        # Convert to format expected by violin plot function
        slopes_data = {k: np.array([v]) for k, v in model_slopes.items()}

    # Create the combined visualization
    output_path = os.path.join(plot_dir, f"{prefix}combined.png")
    result_path = create_combined_dependency_violin_plot(
        model=model_obj, slopes_data=slopes_data, output_path=output_path, dpi=300
    )

    # Also create just the dependency plots for additional use
    dependency_path = os.path.join(plot_dir, f"{prefix}dependencies.png")
    plt.figure(figsize=(18, 6))
    features, _ = plot_paper_style_shap_dependencies(model_obj)
    plt.tight_layout()
    plt.savefig(dependency_path, dpi=300)
    plt.close()

    return result_path


### HELPER FUNCTION FOR LOADING SAVED SLOPES DATA ###
def load_slopes_data(file_path=None, checkpoint_dir="./checkpoints/", use_latest=True):
    """
    Load slopes data from a pickle file created by multi_model_shap_analysis.py

    Arguments:
        file_path: Path to a specific slopes data file
        checkpoint_dir: Directory containing checkpoint files
        use_latest: If True and file_path is None, uses the most recent slopes data file

    Returns:
        Dictionary with feature names as keys and numpy arrays of slope values
    """
    import pickle
    import os
    import glob

    if file_path is not None and os.path.exists(file_path):
        # Use the specified file
        with open(file_path, "rb") as f:
            slopes_data = pickle.load(f)
        print(f"Loaded slopes data from: {file_path}")
        return slopes_data

    # Look for slopes data files in the checkpoint directory
    slopes_files = glob.glob(os.path.join(checkpoint_dir, "slopes_data_*.pkl"))

    if not slopes_files:
        # Check for checkpoint files if no specific slopes data files found
        checkpoint_files = glob.glob(
            os.path.join(checkpoint_dir, "slopes_checkpoint.pkl")
        )
        if checkpoint_files:
            print(
                "Found checkpoint file, but no slopes_data files. Loading from checkpoint..."
            )
            with open(checkpoint_files[0], "rb") as f:
                checkpoint = pickle.load(f)
                if "slopes_collection" in checkpoint:
                    # Convert slopes_collection to numpy arrays if needed
                    slopes_data = {}
                    for feature, values in checkpoint["slopes_collection"].items():
                        if not isinstance(values, np.ndarray):
                            slopes_data[feature] = np.array(values)
                        else:
                            slopes_data[feature] = values
                    print(f"Loaded slopes data from checkpoint: {checkpoint_files[0]}")
                    return slopes_data
        raise FileNotFoundError(f"No slopes data files found in {checkpoint_dir}")

    if use_latest:
        # Use the most recent file
        latest_file = max(slopes_files, key=os.path.getctime)
        with open(latest_file, "rb") as f:
            slopes_data = pickle.load(f)
        print(
            f"Loaded slopes data from the most recent file: {os.path.basename(latest_file)}"
        )
        return slopes_data
    else:
        # Print available files and let the user choose
        print("Available slopes data files:")
        for i, file in enumerate(slopes_files):
            print(f"{i}: {os.path.basename(file)}")

        choice = input(
            "Enter the number of the file to use (or press Enter for the most recent): "
        )
        if choice.strip() == "":
            # Use the most recent file
            file_to_use = max(slopes_files, key=os.path.getctime)
        else:
            # Use the chosen file
            file_to_use = slopes_files[int(choice)]

        with open(file_to_use, "rb") as f:
            slopes_data = pickle.load(f)
        print(f"Loaded slopes data from: {os.path.basename(file_to_use)}")
        return slopes_data


### Example usage ###
if __name__ == "__main__":
    # This code will run when the file is executed directly
    print("Figure 3 module - run with a model to create visualizations")

    # Example of how to use:
    """
    # For a single model:
    from electricity_price_model import ElectricityPriceModel
    model = ElectricityPriceModel()
    # Train model and calculate SHAP values...
    
    # Create just the dependency plots
    features, slopes = plot_paper_style_shap_dependencies(model)
    plt.savefig("dependency_plots.png")
    plt.close()
    
    # Create the full combined figure
    create_combined_dependency_violin_plot(model=model, output_path="figure3_combined.png")
    
    # Load and use saved slopes data from multi_model_shap_analysis.py
    slopes_data = load_slopes_data()  # Automatically finds the most recent file
    
    create_combined_dependency_violin_plot(
        model=model,  # For top dependency plots
        slopes_data=slopes_data,  # For bottom violin plot
        output_path="figure3_with_multi_model_slopes.png"
    )
    """
