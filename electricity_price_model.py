import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
import random


class GermanElectricityDataLoader:
    """
    Data loader for German electricity price prediction datasets.
    Handles loading and preprocessing of multiple CSV files.
    """

    def __init__(self, data_dir):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the CSV files
        """
        self.data_dir = data_dir

    def _safe_convert_to_numeric(self, df, column):
        """
        Safely convert a column to numeric by replacing common missing value indicators
        and using pd.to_numeric with errors='coerce'.
        """
        if column in df.columns:
            # Replace common missing value indicators with NaN
            df[column] = df[column].replace(["n/e", "N/A", "n.e.", "n/a", "-"], np.nan)
            # Convert to numeric, coercing errors to NaN
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    def _ensure_timestamp_datetime(self, df):
        """
        Ensure the timestamp column is properly converted to datetime.
        """
        if "timestamp" in df.columns:
            # Check if timestamp is not already datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                try:
                    # Try to convert to datetime, handling both German and American formats
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                except Exception as e:
                    print(
                        f"Warning: Could not convert timestamp column to datetime: {str(e)}"
                    )

        return df

    def _process_timestamp_field(self, df):
        """
        Helper method to process timestamp fields in various formats.
        """
        # Look for timestamp column with various possible names
        timestamp_col = None
        column_list = list(df.columns)

        for col_name in [
            "timestamp",
            "date",
            "time",
            "MTU",
            "datetime",
            "Time (CET/CEST)",
            "MTU (CET/CEST)",
        ]:
            if col_name in column_list:
                timestamp_col = col_name
                break

        if timestamp_col is not None:
            try:
                # For formats like "01.01.2017 00:00 - 01.01.2017 01:00 (CET/CEST)" or "01.01.2017 00:00 - 01.01.2017 00:15"
                first_value = str(df[timestamp_col].iloc[0]) if len(df) > 0 else ""

                if " - " in first_value:
                    # Extract start time with safer pattern extraction
                    extracted = df[timestamp_col].str.extract(
                        r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})"
                    )
                    if not extracted.empty:
                        df["timestamp"] = pd.to_datetime(
                            extracted.iloc[:, 0],
                            format="%d.%m.%Y %H:%M",
                            errors="coerce",
                        )
                        print(
                            f"Zeitstempelspalte erfolgreich aus '{timestamp_col}' erstellt"
                        )
                else:
                    # Try standard datetime parsing
                    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
                    print(
                        f"Zeitstempelspalte erfolgreich aus '{timestamp_col}' mit Standard-Parsing erstellt"
                    )
            except Exception as e:
                print(
                    f"Warnung: Timestamp-Spalte '{timestamp_col}' konnte nicht geparst werden: {str(e)}"
                )
                if "timestamp" not in column_list:
                    df["timestamp"] = df[timestamp_col]

        # Print debugging info about the timestamp column
        if "timestamp" in df.columns and len(df) > 0:
            print(f"Timestamp-Informationen:")
            print(f"  Datentyp: {df['timestamp'].dtype}")
            print(f"  Gültige Werte: {df['timestamp'].notna().sum()} von {len(df)}")
            if df["timestamp"].notna().any():
                print(f"  Beispiel: {df['timestamp'][df['timestamp'].notna()].iloc[0]}")


    def _process_timestamp_field(self, df):
        """
        Helper method to process timestamp fields in various formats.

        Args:
            df: DataFrame to process
        """
        # Look for timestamp column with various possible names
        timestamp_col = None
        for col_name in [
            "timestamp",
            "date",
            "time",
            "MTU",
            "datetime",
            "Time (CET/CEST)",
        ]:
            if col_name in df.columns:
                timestamp_col = col_name
                break

        if timestamp_col:
            # Try different parsing approaches based on observed formats
            try:
                # For formats like "01.01.2017 00:00 - 01.01.2017 01:00 (CET/CEST)" or "01.01.2017 00:00 - 01.01.2017 00:15"
                if df[timestamp_col].dtype == "object" and " - " in str(
                    df[timestamp_col].iloc[0]
                ):
                    # Extract start time
                    df["timestamp"] = (
                        df[timestamp_col]
                        .str.extract(r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})")
                        .apply(
                            lambda x: (
                                pd.to_datetime(x, format="%d.%m.%Y %H:%M")
                                if pd.notna(x)
                                else pd.NaT
                            )
                        )
                    )
                else:
                    # Try standard datetime parsing
                    df["timestamp"] = pd.to_datetime(df[timestamp_col])
            except:
                # If all else fails, leave it as is
                print(
                    f"Warning: Could not parse timestamp column '{timestamp_col}' in dataframe"
                )
                if "timestamp" not in df.columns:
                    df["timestamp"] = df[timestamp_col]

    def _is_15min_data(self, df):
        """
        Check if the dataframe contains 15-minute interval data.

        Args:
            df: DataFrame to check

        Returns:
            Boolean indicating if data is in 15-minute intervals
        """
        if "timestamp" not in df.columns or len(df) < 2:
            return False

        # Sort by timestamp and get the first time difference
        df_sorted = df.sort_values("timestamp")

        try:
            # Calculate time differences between consecutive timestamps
            time_diffs = df_sorted["timestamp"].diff().dropna()

            if len(time_diffs) == 0:
                return False

            # Get the most common time difference in minutes
            most_common_diff = time_diffs.dt.total_seconds().mode().iloc[0] / 60

            # If the most common difference is around 15 minutes
            return abs(most_common_diff - 15) < 5  # Allow for small deviations
        except:
            return False

    def _resample_to_hourly(self, df, value_columns=None):
        """
        Resample 15-minute data to hourly intervals by averaging.

        Args:
            df: DataFrame with 15-minute data
            value_columns: List of column names containing values to resample
                           If None, will try to identify numeric columns

        Returns:
            Resampled hourly DataFrame
        """
        if "timestamp" not in df.columns:
            print("Warning: Cannot resample data without timestamp column")
            return df

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Ensure timestamp is datetime
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        # If no value columns specified, find numeric columns
        if value_columns is None:
            value_columns = [
                col
                for col in df_copy.columns
                if col != "timestamp" and pd.api.types.is_numeric_dtype(df_copy[col])
            ]

        if not value_columns:
            print("Warning: No numeric columns found for resampling")
            return df_copy

        # Set timestamp as index for resampling
        df_copy = df_copy.set_index("timestamp")

        # Keep only the columns we want to resample
        cols_to_keep = value_columns + [
            col
            for col in df_copy.columns
            if col not in value_columns and col != "timestamp"
        ]
        df_copy = df_copy[cols_to_keep]

        # Resample to hourly frequency, using mean for numeric columns
        resampled = df_copy.resample("h").mean()

        # Reset index to get timestamp back as a column
        resampled = resampled.reset_index()

        return resampled


class ElectricityPriceModel:
    """
    Gradient Boosted Trees model for German electricity price prediction
    with SHAP explanation as described in the scientific paper.
    """

    def __init__(self, random_state=42):
        """Initialize the model with parameters from the paper."""
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def weekly_shuffle_split(self, data, time_column, test_size=0.2, val_size=0.32):
        """
        Split data using weekly shuffle as described in the paper.

        Arguments:
            data: DataFrame with time-series data
            time_column: Column name containing datetime information
            test_size: Proportion for test set (default: 20%)
            val_size: Proportion for validation set (default: 32%)

        Returns:
            train, validation splits, and test DataFrames
        """
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])

        # Extract week information
        data = data.copy()
        data["week"] = data[time_column].dt.isocalendar().week
        data["year"] = data[time_column].dt.year

        # Get unique weeks
        unique_weeks = data[["year", "week"]].drop_duplicates().reset_index(drop=True)

        # Shuffle weeks
        np.random.seed(self.random_state)
        shuffled_weeks = unique_weeks.sample(frac=1, random_state=self.random_state)

        # Calculate split points
        n_weeks = len(shuffled_weeks)
        test_weeks = shuffled_weeks.iloc[: int(n_weeks * test_size)]
        val_weeks = shuffled_weeks.iloc[
            int(n_weeks * test_size) : int(n_weeks * (test_size + val_size))
        ]
        train_weeks = shuffled_weeks.iloc[int(n_weeks * (test_size + val_size)) :]

        # Create masks for data splitting
        test_mask = data.apply(
            lambda x: any(
                (x["year"] == year) & (x["week"] == week)
                for year, week in zip(test_weeks["year"], test_weeks["week"])
            ),
            axis=1,
        )
        val_mask = data.apply(
            lambda x: any(
                (x["year"] == year) & (x["week"] == week)
                for year, week in zip(val_weeks["year"], val_weeks["week"])
            ),
            axis=1,
        )
        train_mask = ~(test_mask | val_mask)

        # Split the data
        train_data = data[train_mask].drop(["week", "year"], axis=1)
        val_data = data[val_mask].drop(["week", "year"], axis=1)
        test_data = data[test_mask].drop(["week", "year"], axis=1)

        # Split validation into 4 parts as mentioned in the paper
        val_data_splits = np.array_split(val_data, 4)

        return train_data, val_data_splits, test_data

    def train(self, data, features=None, target="price", time_column="timestamp"):
        """
        Train the GBT model using LightGBM with the paper's methodology.

        Arguments:
            data: DataFrame with all data
            features: List of feature column names. If None, will use all appropriate columns
            target: Target column name (electricity price)
            time_column: Datetime column name
        """
        # If features not specified, use all appropriate columns based on the paper's feature list
        if features is None:
            # Determine appropriate features based on Fig. 2 from the paper
            potential_features = [
                "load_forecast",  # Load day-ahead (highest importance)
                "wind_forecast",  # Wind day-ahead (essentially tied with load)
                "solar_forecast",  # Solar day-ahead
                "total_generation",  # Scheduled generation total
                "net_import_export",  # Import export day-ahead
                "oil",  # Oil price
                "natural_gas",  # Natural gas price
                "load_forecast_ramp",  # Load ramp day-ahead
                "wind_forecast_ramp",  # Wind ramp day-ahead
                "solar_forecast_ramp",  # Solar ramp day-ahead
                "total_generation_ramp",  # Total generation ramp day-ahead
                "net_import_export_ramp",  # Import export ramp day-ahead
            ]

            # Only include features that actually exist in the dataframe
            features = [f for f in potential_features if f in data.columns]
            print(f"Using features: {features}")

        if not features:
            raise ValueError("No valid features found or provided for training.")
        if target in features:
            features.remove(target)
            print(f"Removed target column '{target}' from features list.")

        self.feature_names = features

        # Split data with weekly shuffle
        train_data, val_data_splits, test_data = self.weekly_shuffle_split(
            data, time_column
        )

        if train_data.empty or target not in train_data.columns:
            raise ValueError(
                "Training data is empty or missing the target column after split."
            )

        # Prepare train, validation and test sets
        X_train = train_data[features]
        y_train = train_data[target]

        X_val_splits = [val_split[features] for val_split in val_data_splits]
        y_val_splits = [val_split[target] for val_split in val_data_splits]

        # --- Store test timestamps HERE ---
        if (
            not test_data.empty
            and target in test_data.columns
            and time_column in test_data.columns
        ):
            X_test = test_data[features]
            y_test = test_data[target]
            # Store test data for later evaluation - includes original index
            self.X_test = X_test
            self.y_test = y_test
            # *** NEW: Store the corresponding timestamps ***
            self.test_timestamps = test_data[time_column]
            print("-" * 20)
            print(f"DEBUG [train]: Assigning self.test_timestamps.")
            print(f"DEBUG [train]: Type = {type(self.test_timestamps)}")
            print(f"DEBUG [train]: Length = {len(self.test_timestamps)}")
            print(f"DEBUG [train]: Is empty? = {self.test_timestamps.empty}")
            if not self.test_timestamps.empty:
                print(f"DEBUG [train]: Head =\n{self.test_timestamps.head()}")
                print(
                    f"DEBUG [train]: Contains NaT? = {self.test_timestamps.isnull().any()}"
                )
            print("-" * 20)
            # *****************************************
        else:
            print(
                "Warning: Test set is empty, missing target, or missing timestamp column. Skipping test set preparation."
            )
            self.X_test = pd.DataFrame(columns=features)
            self.y_test = pd.Series(dtype=float)
            self.test_timestamps = pd.Series(
                dtype="datetime64[ns]"
            )  # Store empty series
        # --- End storing test timestamps ---

        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test

        # Define parameter search space as mentioned in the paper
        param_grid = {
            "num_leaves": [31, 50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 500, 1000],
            "min_child_samples": [10, 20, 50, 100],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
        }

        # Random search for best hyperparameters as mentioned in the paper
        best_models = []
        best_scores = []

        print("Starting random search for hyperparameters...")

        # Using a more compatible approach for LightGBM
        for i in range(30):  # Number of random search iterations
            # Randomly sample hyperparameters
            np.random.seed(self.random_state + i)
            params = {k: np.random.choice(v) for k, v in param_grid.items()}

            # Add fixed parameters
            params.update(
                {
                    "objective": "regression",
                    "metric": "mae",
                    "verbosity": -1,  # Use verbosity instead of verbose
                    "random_state": self.random_state,
                }
            )

            print(f"Training model {i+1}/30 with parameters: {params}")

            # Create LightGBM dataset objects
            train_dataset = lgb.Dataset(X_train, label=y_train)
            val_datasets = [
                lgb.Dataset(X_val, label=y_val)
                for X_val, y_val in zip(X_val_splits, y_val_splits)
            ]

            # Early stopping settings - handle this more simply
            stopping_rounds = 50

            # Train LightGBM model with early stopping
            # Using the updated API (tested with LightGBM versions 3.0+)
            try:
                # First try the newer API with callbacks
                model = lgb.train(
                    params,
                    train_dataset,
                    num_boost_round=1000,  # Maximum number of iterations
                    valid_sets=[train_dataset] + val_datasets,
                    valid_names=["train"]
                    + [f"val_{j}" for j in range(len(val_datasets))],
                    callbacks=[lgb.early_stopping(stopping_rounds, verbose=False)],
                    # verbose_eval=False,
                )
            except TypeError as e:
                # If that fails, try the older API
                print(f"Trying older LightGBM API due to error: {str(e)}")
                params["early_stopping_round"] = stopping_rounds

                model = lgb.train(
                    params,
                    train_dataset,
                    num_boost_round=1000,
                    valid_sets=[train_dataset] + val_datasets,
                    valid_names=["train"]
                    + [f"val_{j}" for j in range(len(val_datasets))],
                )

            # Evaluate on test set
            y_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred)

            print(f"Model {i+1} test MAE: {test_mae:.4f}")

            # Save model and score
            best_models.append(model)
            best_scores.append(test_mae)

        # Select best model
        best_idx = np.argmin(best_scores)
        self.model = best_models[best_idx]
        self.best_score = best_scores[best_idx]

        print(f"Best model selected with test MAE: {self.best_score:.4f}")

        # <<< --- ADD DEBUG --- >>>
        print(
            f"DEBUG [train - end]: hasattr(self, 'test_timestamps') before return? = {hasattr(self, 'test_timestamps')}"
        )
        # <<< --- END DEBUG --- >>>

        return self

    def explain_with_shap(self):
        """
        Generate SHAP explanations for the model as described in the paper.

        Returns:
            self (for method chaining)
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating explanations.")

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values for test set
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(self.X_test)

        return self

    def plot_global_feature_importance(self):
        """
        Plot global feature importance using SHAP values (Fig. 2 in paper).
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(0)

        # Normalize feature importance to match Fig. 2 in the paper
        feature_importance_norm = feature_importance / feature_importance.max()

        # Create sorted indices for feature importance
        sorted_idx = np.argsort(feature_importance_norm)

        # Create DataFrame for plotting
        importance_df = pd.DataFrame(
            {
                "Feature": [self.X_test.columns[i] for i in sorted_idx[::-1]],
                "Importance": feature_importance_norm[sorted_idx[::-1]],
                "Rank": range(1, len(sorted_idx) + 1),
            }
        )

        # Plot
        plt.figure(figsize=(12, 8))
        ax = plt.barh(
            importance_df["Feature"], importance_df["Importance"], color="lightblue"
        )

        # Add ranking numbers above bars (matching Fig. 2 from paper)
        for i, (importance, feature) in enumerate(
            zip(importance_df["Importance"], importance_df["Feature"])
        ):
            plt.text(
                importance + 0.01,
                i,
                str(importance_df["Rank"].iloc[i]),
                va="center",
                fontweight="bold",
            )

        plt.xlabel("Feature Importance")
        plt.title(
            "Feature Importance in the GBT Model for Day-Ahead Electricity Prices"
        )
        plt.tight_layout()

        return importance_df

    # def plot_shap_interaction(self, feature1, feature2):
    #     """
    #     Plot SHAP interaction plots (Fig. 4 and 5 in paper).

    #     Arguments:
    #         feature1: First feature name
    #         feature2: Second feature name
    #     """
    #     if self.shap_values is None:
    #         raise ValueError("SHAP values must be calculated first.")

    #     # Calculate SHAP interaction values (can be computationally expensive)
    #     print("Calculating SHAP interaction values (this may take some time)...")
    #     shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)

    #     # Get feature indices
    #     idx1 = list(self.X_test.columns).index(feature1)
    #     idx2 = list(self.X_test.columns).index(feature2)

    #     # Create interaction plot
    #     plt.figure(figsize=(10, 8))
    #     shap.dependence_plot(
    #         (idx1, idx2),
    #         shap_interaction_values,
    #         self.X_test,
    #         display_features=self.X_test,
    #     )

    def evaluate_model_consistency(
        self, data, features=None, target="price", time_column="timestamp", n_splits=10
    ):
        """
        Evaluate model consistency across multiple weekly splits.

        Arguments:
            data: DataFrame with all data
            features: List of feature columns
            target: Target column name
            time_column: Datetime column name
            n_splits: Number of different splits to try

        Returns:
            List of top models for each split with their SHAP importance
        """
        all_top_models = []

        for i in range(n_splits):
            print(f"\nTraining model with split {i+1}/{n_splits}")

            # Create a new model with different random state
            model = ElectricityPriceModel(random_state=i)

            # Train with different random seed
            model.train(data, features, target, time_column)

            # Analyze with SHAP
            model.explain_with_shap()

            # Calculate global feature importance
            feature_importance = np.abs(model.shap_values).mean(0)
            feature_importance_norm = feature_importance / feature_importance.max()

            # Save results
            all_top_models.append(
                {
                    "model": model.model,
                    "score": model.best_score,
                    "shap_importance": dict(zip(features, feature_importance_norm)),
                    "split_seed": i,
                }
            )

            print(f"Model {i+1} best score: {model.best_score:.4f}")

        return all_top_models

    def predict(self, X):
        """
        Make predictions with the trained model.

        Arguments:
            X: DataFrame with features

        Returns:
            Array of price predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        return self.model.predict(X)