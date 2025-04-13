#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Model SHAP Dependency Slopes Analysis

This script trains multiple ElectricityPriceModel instances with different random seeds,
extracts the actual SHAP dependency slopes from each model, and creates a violin plot
showing the distribution of these slopes across all models.

Features:
- Checkpointing to save progress and resume if interrupted
- Progress tracking
- Support for parallel processing (optional)
- Configurable number of models to train

Usage:
    python multi_model_shap_analysis.py [--models N] [--parallel] [--resume]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pickle
from datetime import datetime
import traceback

# Try to import multiprocessing for parallel execution
try:
    import multiprocessing as mp

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# Define directories
DATA_DIR = "./data/xlsx/"
PLOT_DIR = "./plots/"
CHECKPOINT_DIR = "./checkpoints/"

# Ensure directories exist
for directory in [PLOT_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train multiple models and analyze SHAP dependency slopes"
    )
    parser.add_argument(
        "--models",
        type=int,
        default=100,
        help="Number of models to train (default: 100)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2017, 2018, 2019],
        help="Years to use for training (default: 2017, 2018, 2019)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Frequency of checkpoints (every N models)",
    )
    return parser.parse_args()


def load_checkpoint():
    """Load checkpoint if it exists"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "slopes_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            print(
                f"Checkpoint loaded with {len(checkpoint['completed_seeds'])} completed models"
            )
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")

    return {
        "slopes_collection": {},
        "completed_seeds": set(),
        "start_time": time.time(),
    }


def save_checkpoint(checkpoint):
    """Save checkpoint to disk"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "slopes_checkpoint.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    # Also save as JSON for easier inspection
    json_checkpoint = {
        "slopes_collection": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in checkpoint["slopes_collection"].items()
        },
        "completed_seeds": list(checkpoint["completed_seeds"]),
        "start_time": checkpoint["start_time"],
        "last_update": time.time(),
    }

    json_path = os.path.join(CHECKPOINT_DIR, "slopes_checkpoint.json")
    with open(json_path, "w") as f:
        json.dump(json_checkpoint, f, indent=2)

    print(
        f"Checkpoint saved with {len(checkpoint['completed_seeds'])} completed models"
    )


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
            f"WARNING: None of the required features {required_features} are available in the model!"
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

            # Print debug info
            print(f"\nProcessing {feature} slope calculation:")
            print(f"  Data points: {len(x_values)}")
            print(f"  X range: {x_values.min():.2f} to {x_values.max():.2f}")
            print(f"  SHAP range: {y_values.min():.2f} to {y_values.max():.2f}")

            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)

            print(f"  SLOPE RESULT: {slope:.8f}, r²: {r_value**2:.4f}")

            # Store the slope
            slopes[feature] = slope

        except Exception as e:
            print(f"Error calculating slope for {feature}: {str(e)}")

    # Verify we have results for all required features
    for feature in required_features:
        if feature not in slopes:
            print(f"WARNING: No slope calculated for {feature}")

    return slopes


def train_model_and_extract_slopes(seed, years):
    """
    Train a single model with the given random seed and extract SHAP dependency slopes.

    Arguments:
        seed: Random seed for model training
        years: Years to use for training data

    Returns:
        Dictionary of slopes or None if an error occurred
    """
    try:
        print(f"Training model with seed {seed}")

        # Import necessary modules
        from electricity_price_model import ElectricityPriceModel
        from excel_electricity_loader import ExcelElectricityDataLoader

        # Initialize the data loader
        data_loader = ExcelElectricityDataLoader(DATA_DIR)

        # Load and prepare the dataset
        dataset = data_loader.prepare_full_dataset(years=years)

        # Determine valid features
        numeric_features = dataset.select_dtypes(include=["number"]).columns.tolist()

        # Force inclusion of key features
        must_include = ["load_forecast", "solar_forecast", "wind_forecast"]
        valid_features = [f for f in must_include if f in dataset.columns]

        # Add other numeric features
        for feature in numeric_features:
            if (
                feature != "price"
                and not feature.endswith("_ramp")
                and feature not in valid_features
            ):
                non_null_percent = dataset[feature].notna().sum() / len(dataset) * 100
                if non_null_percent > 50:
                    valid_features.append(feature)
                    # Also include the ramp feature if it exists
                    ramp_feature = f"{feature}_ramp"
                    if ramp_feature in dataset.columns:
                        valid_features.append(ramp_feature)

        print(f"Using features: {valid_features}")

        # Verify key features are present
        for key_feature in must_include:
            if key_feature not in valid_features:
                print(
                    f"WARNING: Key feature '{key_feature}' is missing from the dataset!"
                )
                non_null_count = (
                    dataset[key_feature].notna().sum()
                    if key_feature in dataset.columns
                    else 0
                )
                total_count = len(dataset)
                print(
                    f"  Data quality: {non_null_count}/{total_count} non-null values ({non_null_count/total_count*100:.2f}%)"
                )

        # Initialize and train the model
        model = ElectricityPriceModel(random_state=seed)
        model.train(dataset, features=valid_features)

        # Generate SHAP explanations
        model.explain_with_shap()

        # Calculate and return slopes using fixed features
        slopes = calculate_shap_dependency_slopes(model, required_features=must_include)
        return slopes

    except Exception as e:
        print(f"Error training model with seed {seed}: {str(e)}")
        traceback.print_exc()
        return None


def create_violin_plot(slopes_collection, output_path):
    """
    Create a violin plot of SHAP dependency slopes for only load_forecast, solar_forecast, and wind_forecast.

    Arguments:
        slopes_collection: Dictionary mapping feature names to lists of slope values
        output_path: Path to save the plot

    Returns:
        Path to the saved plot
    """
    # Map internal feature names to display names
    feature_display_names = {
        "load_forecast": "Load day-ahead",
        "solar_forecast": "Solar day-ahead",
        "wind_forecast": "Wind day-ahead",
    }

    # Filter to include only the three specified features
    filtered_features = ["load_forecast", "solar_forecast", "wind_forecast"]

    # Keep only features that exist in the slopes_collection
    feature_names = [name for name in filtered_features if name in slopes_collection]

    # If none of the requested features are available, print a warning
    if not feature_names:
        print(
            "Warning: None of the specified features (load_forecast, solar_forecast, wind_forecast) found in data!"
        )
        # Fall back to all features if none of the specified ones exist
        feature_names = list(slopes_collection.keys())

    feature_values = [np.array(slopes_collection[name]) for name in feature_names]
    positions = list(range(1, len(feature_names) + 1))

    # Create figure
    plt.figure(figsize=(10, 6))

    # Set colors to match the reference image - one color per feature
    colors = ["#D5F0F2", "#65B1C1", "#3D629B"]

    # Create violin plot
    violin_parts = plt.violinplot(
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

    # Add box plots inside the violins
    plt.boxplot(
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
    plt.xticks(
        positions,
        [feature_display_names.get(name, name) for name in feature_names],
        fontsize=12,
    )
    plt.ylabel("Slope [EUR MWh$^{-2}$]", fontsize=12)
    plt.title("d", loc="left", fontweight="bold", fontsize=14)

    # Set y-axis limits based on data
    y_min = min([np.min(vals) for vals in feature_values])
    y_max = max([np.max(vals) for vals in feature_values])
    margin = (y_max - y_min) * 0.1
    plt.ylim(y_min - margin, y_max + margin)

    # Add grid lines
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Lighten the border
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#888888")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Violin plot created with {len(feature_names)} features: {', '.join(feature_names)}"
    )

    return output_path

## Claude version anhand des papers
# def create_violin_plot(slopes_collection, output_path):
#     """
#     Create a violin plot of SHAP dependency slopes matching the style from the reference paper.
    
#     Arguments:
#         slopes_collection: Dictionary mapping feature names to lists of slope values
#         output_path: Path to save the plot
        
#     Returns:
#         Path to the saved plot
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Map internal feature names to display names
#     feature_display_names = {
#         'load_forecast': 'Load day-ahead',
#         'solar_forecast': 'Solar day-ahead',
#         'wind_forecast': 'Wind day-ahead'
#     }
    
#     # Specified features in the exact order from the paper
#     filtered_features = ['load_forecast', 'solar_forecast', 'wind_forecast']
    
#     # Keep only features that exist in the slopes_collection
#     feature_names = [name for name in filtered_features if name in slopes_collection]
    
#     # If none of the requested features are available, print a warning
#     if not feature_names:
#         print("Warning: None of the specified features (load_forecast, solar_forecast, wind_forecast) found in data!")
#         return None
    
#     # Get values and possibly negate renewable generation slopes as mentioned in the paper
#     # "we simplify the comparison of the three features by multiplying the renewable generations by −1"
#     feature_values = []
#     for name in feature_names:
#         values = np.array(slopes_collection[name])
#         # Check if renewable feature (solar or wind) AND if values are primarily negative
#         if name in ['solar_forecast', 'wind_forecast'] and np.median(values) < 0:
#             values = -values  # Negate values for renewable generation as described in the paper
#             print(f"Negating values for {name} as per paper methodology")
#         feature_values.append(values)
    
#     positions = list(range(1, len(feature_names) + 1))
    
#     # Create figure with specific size to match paper
#     plt.figure(figsize=(12, 5))
    
#     # Set colors to exactly match the reference image
#     colors = ['#D5F0F2', '#65B1C1', '#3D629B']  # Light blue, medium blue, dark blue
    
#     # Create violin plot
#     violin_parts = plt.violinplot(
#         feature_values,
#         positions=positions,
#         showmeans=False, 
#         showmedians=False,
#         showextrema=False
#     )
    
#     # Customize violin appearance
#     for i, pc in enumerate(violin_parts['bodies']):
#         pc.set_facecolor(colors[i % len(colors)])
#         pc.set_edgecolor('black')
#         pc.set_alpha(1)
    
#     # Add box plots inside the violins - styled to match the paper
#     boxplots = plt.boxplot(
#         feature_values,
#         positions=positions,
#         widths=0.15,
#         patch_artist=False,
#         boxprops=dict(linestyle='-', linewidth=1.5, color='black'),
#         whiskerprops=dict(linestyle='-', linewidth=1.5, color='black'),
#         medianprops=dict(linestyle='-', linewidth=1.5, color='black'),
#         capprops=dict(linestyle='-', linewidth=1.5, color='black'),
#         flierprops=dict(marker='.', markerfacecolor='black', markersize=3)
#     )
    
#     # Customize the plot
#     plt.xticks(positions, [feature_display_names.get(name, name) for name in feature_names], fontsize=12)
#     plt.ylabel('Slope [EUR MWh$^{-2}$]', fontsize=12)  # Using LaTeX for superscript
    
#     # Add the "d" label in top-left corner as in the paper
#     plt.title('d', loc='left', fontweight='bold', fontsize=14)
    
#     # Add grid lines (light gray, dashed)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
#     # Set y-axis limits to match data range with a small margin
#     y_min = min([np.min(vals) for vals in feature_values])
#     y_max = max([np.max(vals) for vals in feature_values])
#     margin = (y_max - y_min) * 0.1
#     plt.ylim(y_min - margin, y_max + margin)
    
#     # Make the plot background white and lighten the border
#     plt.gca().set_facecolor('white')
#     for spine in plt.gca().spines.values():
#         spine.set_edgecolor('#888888')
    
#     # Save figure with high resolution
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"Violin plot saved to: {output_path}")
    
#     # Create a version with scientific notation if needed
#     plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
#     sci_output_path = output_path.replace('.png', '_scientific.png')
#     plt.savefig(sci_output_path, dpi=300, bbox_inches='tight')
#     print(f"Scientific notation version saved to: {sci_output_path}")
    
#     plt.close()
    
#     print(f"Violin plot created with {len(feature_names)} features: {', '.join(feature_names)}")
    
#     return output_path


def process_model(args):
    """Process a single model for parallel execution"""
    seed, years = args
    return seed, train_model_and_extract_slopes(seed, years)


def main():
    """Main function to run the analysis"""
    args = parse_arguments()

    # Check if parallel processing is requested but not available
    if args.parallel and not PARALLEL_AVAILABLE:
        print(
            "Warning: Parallel processing requested but multiprocessing module is not available."
        )
        print("Falling back to sequential processing.")
        args.parallel = False

    # Load checkpoint or initialize new one
    if args.resume:
        checkpoint = load_checkpoint()
    else:
        checkpoint = {
            "slopes_collection": {},
            "completed_seeds": set(),
            "start_time": time.time(),
        }

    # Determine which seeds to process
    total_models = args.models
    all_seeds = list(range(total_models))
    seeds_to_process = [
        seed for seed in all_seeds if seed not in checkpoint["completed_seeds"]
    ]

    print(f"Will process {len(seeds_to_process)} out of {total_models} models")

    if not seeds_to_process:
        print("All models already processed. Generating final violin plot.")
    else:
        # Process models
        if args.parallel:
            print(f"Using parallel processing with {mp.cpu_count()} cores")
            with mp.Pool() as pool:
                for i, (seed, slopes) in enumerate(
                    pool.imap_unordered(
                        process_model, [(seed, args.years) for seed in seeds_to_process]
                    )
                ):

                    if slopes is not None:
                        # Initialize collection for new features
                        for feature in slopes:
                            if feature not in checkpoint["slopes_collection"]:
                                checkpoint["slopes_collection"][feature] = []

                            # Append the slope
                            checkpoint["slopes_collection"][feature].append(
                                slopes[feature]
                            )

                        # Mark as completed
                        checkpoint["completed_seeds"].add(seed)

                    # Print progress
                    completed = len(checkpoint["completed_seeds"])
                    print(
                        f"Progress: {completed}/{total_models} models ({completed/total_models*100:.1f}%)"
                    )

                    # Save checkpoint periodically
                    if completed % args.checkpoint_freq == 0:
                        save_checkpoint(checkpoint)
        else:
            # Sequential processing
            for i, seed in enumerate(seeds_to_process):
                slopes = train_model_and_extract_slopes(seed, args.years)

                if slopes is not None:
                    # Initialize collection for new features
                    for feature in slopes:
                        if feature not in checkpoint["slopes_collection"]:
                            checkpoint["slopes_collection"][feature] = []

                        # Append the slope
                        checkpoint["slopes_collection"][feature].append(slopes[feature])

                    # Mark as completed
                    checkpoint["completed_seeds"].add(seed)

                # Print progress
                completed = len(checkpoint["completed_seeds"])
                print(
                    f"Progress: {completed}/{total_models} models ({completed/total_models*100:.1f}%)"
                )

                # Save checkpoint periodically
                if completed % args.checkpoint_freq == 0:
                    save_checkpoint(checkpoint)

        # Save final checkpoint
        save_checkpoint(checkpoint)
    
    # Convert lists to numpy arrays for final analysis
    slopes_collection_np = {}
    for feature, slopes in checkpoint["slopes_collection"].items():
        slopes_collection_np[feature] = np.array(slopes)

    # Print summary statistics
    print("\nSlope summary statistics:")
    for feature, slopes in slopes_collection_np.items():
        print(
            f"  {feature}: count={len(slopes)}, mean={np.mean(slopes):.6f}, "
            f"std={np.std(slopes):.6f}, min={np.min(slopes):.6f}, max={np.max(slopes):.6f}"
        )

    # Create the violin plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(PLOT_DIR, f"shap_dependency_slopes_{timestamp}.png")
    final_path = create_violin_plot(slopes_collection_np, output_path)

    print(f"\nViolin plot saved to: {final_path}")

    # Save data for future reference
    data_path = os.path.join(CHECKPOINT_DIR, f"slopes_data_{timestamp}.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(slopes_collection_np, f)

    print(f"Slope data saved to: {data_path}")

    # Calculate elapsed time
    elapsed = time.time() - checkpoint["start_time"]
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
