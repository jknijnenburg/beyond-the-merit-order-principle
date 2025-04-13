# %%
#!/usr/bin/env python3
"""
Complete workflow for electricity price forecasting:
1. Process and fix fuel price data
2. Train the model with proper features
3. Generate visualizations and metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime
import traceback
import plots

# Set your data directory
DATA_DIR = "./data/xlsx/"
PLOT_DIR = "./plots/"


def process_fuel_prices():
    """Process fuel prices and create hourly interpolated data"""
    # Import the direct fuel price loader
    from fuel_price_loader import (
        load_oil_prices,
        load_gas_prices,
        create_hourly_fuel_prices,
    )

    print("=== STEP 1: PROCESSING FUEL PRICE DATA ===\n")

    # Load the raw data
    oil_df = load_oil_prices()
    gas_df = load_gas_prices()

    # Create hourly fuel prices
    hourly_df = create_hourly_fuel_prices(oil_df, gas_df)

    if hourly_df is not None:
        print("\nFuel price processing complete!")
        print(f"Created hourly fuel price dataset with {len(hourly_df)} rows")
        return hourly_df
    else:
        print("\nFailed to create hourly fuel prices")
        return None


def run_model(fuel_prices_df=None):
    """Run the electricity price forecasting model"""
    # Define the time column name
    time_column = "timestamp"

    print("\n=== STEP 2: RUNNING ELECTRICITY PRICE MODEL ===\n")

    # Add the current directory to the path so we can import the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    try:
        # Import the model modules
        from excel_electricity_loader import ExcelElectricityDataLoader
        from electricity_price_model import ElectricityPriceModel
        from create_fig_4_paper_style_plot import extend_model_with_fixed_interactions

        # Initialize the Excel-based data loader
        print("Initializing Excel data loader...")
        data_loader = ExcelElectricityDataLoader(DATA_DIR)

        # Load and prepare the dataset
        print("\nPreparing dataset from Excel files...")
        dataset = data_loader.prepare_full_dataset(years=[2017, 2018, 2019])

        # Merge with the fuel prices if available
        if fuel_prices_df is not None:
            print("\nMerging hourly fuel prices into the dataset...")
            # Make sure timestamps are datetime objects
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
            fuel_prices_df["timestamp"] = pd.to_datetime(fuel_prices_df["timestamp"])

            if "oil" in dataset.columns:
                print(
                    f"Removing existing 'oil' column with {dataset['oil'].notna().sum()} non-null values"
                )
                dataset = dataset.drop(columns=["oil"])

            if "natural_gas" in dataset.columns:
                print(
                    f"Removing existing 'natural_gas' column with {dataset['natural_gas'].notna().sum()} non-null values"
                )
                dataset = dataset.drop(columns=["natural_gas"])

            # Merge on timestamp
            original_rows = len(dataset)
            dataset = pd.merge(dataset, fuel_prices_df, on="timestamp", how="left")
            print(f"Merged dataset now has {len(dataset)} rows (was {original_rows})")

            # Check the fuel price columns
            for col in ["oil", "natural_gas"]:
                if col in dataset.columns:
                    non_null = dataset[col].notna().sum()
                    percent = non_null / len(dataset) * 100
                    print(f"  {col}: {non_null} non-null values ({percent:.2f}%)")

                    # Sample values
                    if non_null > 0:
                        print(
                            f"  Sample {col} values: {dataset.loc[dataset[col].notna(), col].head(3).tolist()}"
                        )

        print(f"\nDataset successfully loaded with {len(dataset)} data points")
        print(
            f"Time period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}"
        )
        print(f"Available features: {dataset.columns.tolist()}")

        # Save the prepared dataset for inspection
        final_dataset_path = os.path.join(
            DATA_DIR,
            f"prepared_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        dataset.to_excel(final_dataset_path, index=False)
        print(f"Saved prepared dataset to {final_dataset_path}")

        # Check which features have data
        print("\nAnalyzing feature data availability:")
        numeric_features = dataset.select_dtypes(include=["number"]).columns.tolist()
        valid_features = []

        for feature in numeric_features:
            if feature != "price" and not feature.endswith(
                "_ramp"
            ):  # Skip price and derived ramp features
                non_null_count = dataset[feature].notna().sum()
                non_null_percent = non_null_count / len(dataset) * 100
                print(
                    f"  {feature}: {non_null_count} non-null values ({non_null_percent:.2f}%)"
                )

                # Only include features with significant data
                if non_null_percent > 50:
                    valid_features.append(feature)
                    # Also include the ramp feature if it exists
                    ramp_feature = f"{feature}_ramp"
                    if ramp_feature in dataset.columns:
                        valid_features.append(ramp_feature)

        print(
            f"\nUsing {len(valid_features)} valid features for training: {valid_features}"
        )

        # Check if we have enough data to train the model
        if len(dataset) < 100:
            print(
                f"Warning: Dataset only has {len(dataset)} rows, which may not be enough for training."
            )
            print("Check the timestamps in your Excel files and try again.")
            return

        # Extend the model class
        ElectricityPriceModel = extend_model_with_fixed_interactions(
            ElectricityPriceModel
        )

        # Initialize and train the model
        print("\nTraining electricity price model...")
        model = ElectricityPriceModel(random_state=42)

        model.train(dataset, features=valid_features)
        print("Model training completed successfully")

        # Generate SHAP explanations
        print("\nGenerating SHAP explanations...")
        model.explain_with_shap()

        # Calculate interaction values
        model.calculate_interaction_values()

        # Create visualizations
        print("\nCreating visualizations...")

        # Feature importance plot
        importance_df = model.plot_global_feature_importance()
        plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"))
        print("Feature importance plot saved to 'feature_importance.png'")

        print("\nFeature importance ranking:")
        for _, row in importance_df.iterrows():
            print(f"  {row['Rank']}. {row['Feature']}: {row['Importance']:.3f}")

        # SHAP dependency plots
        top_features = model.plot_shap_dependency()
        plt.savefig(os.path.join(PLOT_DIR, "shap_dependency.png"))
        print("SHAP dependency plot saved to 'shap_dependency.png'")

        # Paper-Style SHAP-Abhängigkeitsplots für die 3-Features
        top_features = model.plot_paper_style_shap_dependencies(top_n=3)
        plt.savefig(
            os.path.join(PLOT_DIR, "paper_style_shap_dependencies.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Paper-Style SHAP-Abhängigkeitsplots gespeichert als 'paper_style_shap_dependencies.png'"
        )

        # SHAP interaction plot
        # if len(top_features) >= 2:
        #     model.plot_shap_interaction(top_features[0], top_features[1])
        #     plt.savefig(os.path.join(PLOT_DIR, "shap_interaction.png"))
        #     print(
        #         f"SHAP interaction plot between {top_features[0]} and {top_features[1]} saved to 'shap_interaction.png'"
        #     )

        plots.extend_electricity_price_model()

        # slopes = model.calculate_shap_dependency_slopes()
        # model.plot_shap_dependency_slopes(PLOT_DIR, "shap_dependency_slopes.png")

        # Create the comprehensive grid plot with TRUE interaction values
        model.plot_interaction_grid()
        model.plot_single_interaction("load_forecast", "wind_forecast")
        model.plot_single_interaction("net_import_export", "wind_forecast")

        # Evaluate model performance
        print("\nEvaluating model performance...")
        y_pred = model.predict(model.X_test)
        y_true = model.y_test

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = (
            np.mean(np.abs((y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true)) / 2)))
            * 100
        )
        r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")

        print(f"Model performance metrics:")
        print(f"  MAE: {mae:.2f} EUR/MWh")
        print(f"  RMSE: {rmse:.2f} EUR/MWh")
        print(f"  MAPE: {mape:.2f} %")
        print(f" SMAPE: {smape: .2f} %")
        print(f" R^2-score: {r2: .2f}")

        # Plot actual vs. predicted prices
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
        plt.xlabel("Actual Price (EUR/MWh)")
        plt.ylabel("Predicted Price (EUR/MWh)")
        plt.title("Actual vs. Predicted Electricity Prices")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"))
        print("Plot of actual vs. predicted prices saved to 'actual_vs_predicted.png'")

        # time based plot
        simple_plot_path = os.path.join(PLOT_DIR, "actual_vs_predicted_main_range.png")
        plots.create_timebased_plot(model, y_true, y_pred, simple_plot_path)

        # Simple time series plot (no timestamps needed)
        simple_plot_path = os.path.join(PLOT_DIR, "price_comparison_plot.png")
        plots.create_simple_time_series_plot(y_true, y_pred, simple_plot_path)

        print("\nModel training and evaluation completed!")

    except Exception as e:
        print(f"Error running model: {str(e)}")
        traceback.print_exc()


def main():
    # Step 1: Process fuel prices
    fuel_prices_df = process_fuel_prices()

    # Step 2: Run the model with the processed fuel prices
    run_model(fuel_prices_df)


if __name__ == "__main__":
    main()
