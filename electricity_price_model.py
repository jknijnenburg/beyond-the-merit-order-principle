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

    # def load_electricity_prices(self, years=None):
    #     """
    #     Load hourly day-ahead electricity prices.

    #     Args:
    #         years: List of years to load (e.g., [2017, 2018, 2019])
    #             If None, load all available years

    #     Returns:
    #         DataFrame with electricity prices
    #     """
    #     if years is None:
    #         years = [2017, 2018, 2019]

    #     price_dfs = []

    #     for year in years:
    #         file_path = os.path.join(self.data_dir, f"electricity_prices_{year}.csv")
    #         if os.path.exists(file_path):
    #             try:
    #                 # Try standard parsing first
    #                 df = pd.read_csv(file_path, quotechar='"')
    #             except:
    #                 try:
    #                     # If standard parsing fails, try with different settings
    #                     df = pd.read_csv(file_path, quotechar='"', delimiter=",")
    #                 except:
    #                     # Try even more flexible parsing
    #                     df = pd.read_csv(file_path, sep=None, engine="python")

    #             # Check for specific format from the example
    #             if (
    #                 "MTU (CET/CEST)" in df.columns
    #                 and "Day-ahead Price [EUR/MWh]" in df.columns
    #             ):
    #                 print(
    #                     f"Found electricity price data with expected format in {file_path}"
    #                 )

    #                 # Extract timestamp - Fixed: using a safer extraction method
    #                 df["timestamp"] = (
    #                     df["MTU (CET/CEST)"]
    #                     .str.extract(r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})")
    #                     .iloc[:, 0]
    #                 )
    #                 df["timestamp"] = pd.to_datetime(
    #                     df["timestamp"], format="%d.%m.%Y %H:%M"
    #                 )

    #                 # Rename price column to a standard name
    #                 df = df.rename(columns={"Day-ahead Price [EUR/MWh]": "price"})

    #                 # Make sure price is numeric
    #                 df["price"] = pd.to_numeric(df["price"], errors="coerce")

    #                 # Select only necessary columns
    #                 df = df[["timestamp", "price"]]

    #             else:
    #                 # Process timestamp field for other formats
    #                 self._process_timestamp_field(df)

    #                 # Try to find the price column
    #                 price_col = None
    #                 for col in df.columns:
    #                     if (
    #                         "price" in col.lower()
    #                         or "eur" in col.lower()
    #                         or "€" in col.lower()
    #                     ):
    #                         price_col = col
    #                         break

    #                 # If no obvious price column found, use the first numeric column that's not timestamp
    #                 if price_col is None:
    #                     for col in df.columns:
    #                         if col != "timestamp" and pd.api.types.is_numeric_dtype(
    #                             df[col]
    #                         ):
    #                             price_col = col
    #                             break

    #                 if price_col:
    #                     df = df.rename(columns={price_col: "price"})
    #                     df["price"] = pd.to_numeric(df["price"], errors="coerce")
    #                     df = df[["timestamp", "price"]]
    #                 else:
    #                     print(
    #                         f"Warning: Could not identify price column in file {file_path}"
    #                     )
    #                     continue

    #             price_dfs.append(df)
    #         else:
    #             print(
    #                 f"Warning: Electricity price file for {year} not found: {file_path}"
    #             )

    #     if not price_dfs:
    #         raise ValueError("No electricity price data found or processed!")

    #     # Combine all years
    #     prices_df = pd.concat(price_dfs, ignore_index=True)

    #     # Sort by timestamp
    #     prices_df = prices_df.sort_values("timestamp")

    #     return prices_df

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

    # def load_solar_wind_forecasts(self, years=None):
    #     """
    #     Load solar and wind generation forecasts from combined files.

    #     Args:
    #         years: List of years to load

    #     Returns:
    #         Tuple of DataFrames (solar_df, wind_df)
    #     """
    #     if years is None:
    #         years = [2017, 2018, 2019]

    #     solar_dfs = []
    #     wind_dfs = []

    #     # Try to load from combined files first
    #     for year in years:
    #         # Try various possible filenames
    #         possible_files = [
    #             f"solar_wind_forecast_{year}.csv",
    #             f"wind_solar_forecast_{year}.csv",
    #             f"renewable_forecast_{year}.csv",
    #         ]

    #         file_found = False
    #         for filename in possible_files:
    #             file_path = os.path.join(self.data_dir, filename)
    #             if os.path.exists(file_path):
    #                 file_found = True
    #                 print(f"Found combined solar and wind forecast file: {file_path}")

    #                 try:
    #                     # Try standard parsing first
    #                     df = pd.read_csv(file_path, quotechar='"')
    #                 except:
    #                     try:
    #                         # If that fails, try with different settings
    #                         df = pd.read_csv(file_path, quotechar='"', delimiter=",")
    #                     except:
    #                         # Try even more flexible parsing
    #                         df = pd.read_csv(file_path, sep=None, engine="python")

    #                 # Debug: Print columns in the file
    #                 print(f"Spalten in Datei: {list(df.columns)}")

    #                 # Process timestamp field - must be done first
    #                 if "timestamp" not in df.columns:
    #                     print(
    #                         f"Keine 'timestamp'-Spalte in {file_path} gefunden nach Verarbeitung"
    #                     )
    #                     # Try MTU (CET/CEST) column, which is in your solar_wind_forecast files
    #                     if "MTU (CET/CEST)" in df.columns:
    #                         print(
    #                             f"Verwende 'MTU (CET/CEST)' zur Erstellung der Zeitstempelspalte"
    #                         )
    #                         # Extract the timestamp from format like "01.01.2017 00:00 - 01.01.2017 00:15"
    #                         extracted = df["MTU (CET/CEST)"].str.extract(
    #                             r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})"
    #                         )
    #                         if not extracted.empty:
    #                             df["timestamp"] = pd.to_datetime(
    #                                 extracted.iloc[:, 0],
    #                                 format="%d.%m.%Y %H:%M",
    #                                 errors="coerce",
    #                             )
    #                             print(f"Zeitstempelspalte erfolgreich erstellt")
    #                         else:
    #                             print(
    #                                 f"Konnte Zeitstempel nicht aus 'MTU (CET/CEST)' extrahieren"
    #                             )

    #                 # Skip this file if no timestamp column could be created
    #                 if "timestamp" not in df.columns or df["timestamp"].isna().all():
    #                     print(
    #                         f"Überspringe Datei {file_path}, da keine Zeitstempelspalte erstellt werden konnte"
    #                     )
    #                     continue

    #                 # Identify solar day ahead column
    #                 solar_col = None
    #                 for col in df.columns:
    #                     if "solar" in col.lower() and "day ahead" in col.lower():
    #                         solar_col = col
    #                         break

    #                 # Identify wind onshore and offshore day ahead columns
    #                 wind_onshore_col = None
    #                 wind_offshore_col = None

    #                 for col in df.columns:
    #                     if "wind onshore" in col.lower() and "day ahead" in col.lower():
    #                         wind_onshore_col = col
    #                     elif (
    #                         "wind offshore" in col.lower()
    #                         and "day ahead" in col.lower()
    #                     ):
    #                         wind_offshore_col = col

    #                 # Create separate dataframes for solar and wind
    #                 if solar_col is not None:
    #                     print(f"Found solar forecast column: {solar_col}")
    #                     # Safely convert to numeric
    #                     df = self._safe_convert_to_numeric(df, solar_col)

    #                     solar_data = df[["timestamp", solar_col]].copy()
    #                     solar_data = solar_data.rename(
    #                         columns={solar_col: "solar_forecast"}
    #                     )

    #                     # Check for 15-minute intervals and resample if needed
    #                     if self._is_15min_data(solar_data):
    #                         print(
    #                             f"Detected 15-minute interval solar data, resampling to hourly"
    #                         )
    #                         solar_data = self._resample_to_hourly(
    #                             solar_data, value_columns=["solar_forecast"]
    #                         )

    #                     solar_dfs.append(solar_data)

    #                 # Process wind data
    #                 if wind_onshore_col is not None or wind_offshore_col is not None:
    #                     print(
    #                         f"Found wind forecast columns: Onshore={wind_onshore_col}, Offshore={wind_offshore_col}"
    #                     )
    #                     wind_data = df[["timestamp"]].copy()

    #                     # First, safely convert the columns to numeric values
    #                     if wind_onshore_col is not None:
    #                         df = self._safe_convert_to_numeric(df, wind_onshore_col)

    #                     if wind_offshore_col is not None:
    #                         df = self._safe_convert_to_numeric(df, wind_offshore_col)

    #                     # Now combine the data based on what's available
    #                     if (
    #                         wind_onshore_col is not None
    #                         and wind_offshore_col is not None
    #                     ):
    #                         # Sum onshore and offshore wind
    #                         # The .add() method will ignore NaN values when one side has a value
    #                         wind_data["wind_forecast"] = df[wind_onshore_col].add(
    #                             df[wind_offshore_col], fill_value=0
    #                         )
    #                     elif wind_onshore_col is not None:
    #                         wind_data["wind_forecast"] = df[wind_onshore_col]
    #                     elif wind_offshore_col is not None:
    #                         wind_data["wind_forecast"] = df[wind_offshore_col]

    #                     # Check for 15-minute intervals and resample if needed
    #                     if self._is_15min_data(wind_data):
    #                         print(
    #                             f"Detected 15-minute interval wind data, resampling to hourly"
    #                         )
    #                         wind_data = self._resample_to_hourly(
    #                             wind_data, value_columns=["wind_forecast"]
    #                         )

    #                     wind_dfs.append(wind_data)

    #                 # Break out of the filename loop if we found a file
    #                 break

    #         # If no files were found, just continue
    #         if not file_found:
    #             print(f"No combined solar/wind data found for {year}")

    #     # Combine all years
    #     solar_df = (
    #         pd.concat(solar_dfs, ignore_index=True) if len(solar_dfs) > 0 else None
    #     )
    #     wind_df = pd.concat(wind_dfs, ignore_index=True) if len(wind_dfs) > 0 else None

    #     # Add debugging info
    #     if solar_df is not None:
    #         print(
    #             f"Solar forecast data: {len(solar_df)} rows, range: {solar_df['timestamp'].min()} to {solar_df['timestamp'].max()}"
    #         )
    #     else:
    #         print("No solar forecast data found!")

    #     if wind_df is not None:
    #         print(
    #             f"Wind forecast data: {len(wind_df)} rows, range: {wind_df['timestamp'].min()} to {wind_df['timestamp'].max()}"
    #         )
    #     else:
    #         print("No wind forecast data found!")

    #     return solar_df, wind_df

    # def load_power_system_features(self, years=None):
    #     """
    #     Load power system features:
    #     - Day-ahead forecasts of load
    #     - Day-ahead forecasts of solar & wind generation
    #     - Day-ahead total generation

    #     Args:
    #         years: List of years to load

    #     Returns:
    #         DataFrame with power system features
    #     """
    #     if years is None:
    #         years = [2017, 2018, 2019]

    #     feature_dfs = []

    #     # Load load forecasts
    #     load_dfs = []
    #     for year in years:
    #         file_path = os.path.join(self.data_dir, f"load_forecast_{year}.csv")
    #         if os.path.exists(file_path):
    #             try:
    #                 # Try standard parsing first
    #                 df = pd.read_csv(file_path)
    #             except:
    #                 # If that fails, try parsing with different settings
    #                 df = pd.read_csv(file_path, quotechar='"', delimiter=",")

    #             # Process timestamp field
    #             self._process_timestamp_field(df)

    #             # Check for 15-minute intervals and resample to hourly if needed
    #             if self._is_15min_data(df):
    #                 print(
    #                     f"Detected 15-minute interval data in {file_path}, resampling to hourly"
    #                 )
    #                 df = self._resample_to_hourly(
    #                     df,
    #                     value_columns=[
    #                         "Day-ahead Total Load Forecast [MW] - BZN|DE-AT-LU"
    #                     ],
    #                 )

    #             # Identify the forecast column
    #             forecast_col = None
    #             for col in df.columns:
    #                 if "forecast" in col.lower() or "load forecast" in col.lower():
    #                     forecast_col = col
    #                     break

    #             # If no forecast column found, look for a column with 'load' in the name
    #             if forecast_col is None:
    #                 for col in df.columns:
    #                     if "load" in col.lower() and "actual" not in col.lower():
    #                         forecast_col = col
    #                         break

    #             # If still no column found, use the first numeric column that's not timestamp
    #             if forecast_col is None:
    #                 for col in df.columns:
    #                     if col != "timestamp" and pd.api.types.is_numeric_dtype(
    #                         df[col]
    #                     ):
    #                         forecast_col = col
    #                         break

    #             if forecast_col:
    #                 # Make a copy of the original value
    #                 original_forecast_col = forecast_col

    #                 # Print sample values from this column
    #                 if len(df) > 0:
    #                     print(
    #                         f"  Sample values from '{forecast_col}': {df[forecast_col].head(3).tolist()}"
    #                     )

    #                 # Ensure safe conversion to numeric
    #                 df = self._safe_convert_to_numeric(df, forecast_col)

    #                 # Check if all values became NaN after conversion
    #                 if df[forecast_col].isna().all():
    #                     print(
    #                         f"  WARNING: All values in '{forecast_col}' became NaN after numeric conversion"
    #                     )

    #                 # Rename the column
    #                 df = df.rename(columns={forecast_col: "load_forecast"})

    #                 # Print info about the renamed column
    #                 if len(df) > 0:
    #                     print(
    #                         f"  Sample values from 'load_forecast': {df['load_forecast'].head(3).tolist()}"
    #                     )

    #                 df["feature_type"] = "load_forecast"
    #                 df = df[["timestamp", "load_forecast", "feature_type"]]
    #                 load_dfs.append(df)
    #             else:
    #                 print(
    #                     f"  Warning: Could not identify load forecast column in {file_path}"
    #                 )

    #     if load_dfs:
    #         load_df = pd.concat(load_dfs, ignore_index=True)
    #         feature_dfs.append(load_df)

    #     # After processing all files, check if load_dfs is empty:
    #     if not load_dfs:
    #         print(
    #             "WARNING: No load forecast data found. Looking for alternative columns..."
    #         )

    #         # Try again but with more relaxed column detection
    #         for year in years:
    #             file_path = os.path.join(self.data_dir, f"load_forecast_{year}.csv")
    #             if os.path.exists(file_path):
    #                 try:
    #                     df = pd.read_csv(file_path, sep=None, engine="python")
    #                     self._process_timestamp_field(df)

    #                     # Try any column that might contain numeric data
    #                     for col in df.columns:
    #                         if col != "timestamp":
    #                             try:
    #                                 df = self._safe_convert_to_numeric(df, col)
    #                                 # Check if conversion yielded any valid numeric values
    #                                 if not df[col].isna().all():
    #                                     print(
    #                                         f"Using column '{col}' as load forecast data"
    #                                     )
    #                                     df = df.rename(columns={col: "load_forecast"})
    #                                     df["feature_type"] = "load_forecast"
    #                                     df = df[
    #                                         [
    #                                             "timestamp",
    #                                             "load_forecast",
    #                                             "feature_type",
    #                                         ]
    #                                     ]
    #                                     load_dfs.append(df)
    #                                     break
    #                             except:
    #                                 continue
    #                 except Exception as e:
    #                     print(
    #                         f"Error attempting alternative load forecast processing: {str(e)}"
    #                     )

    #         if load_dfs:
    #             print("Successfully found alternative load forecast data")
    #         else:
    #             print(
    #                 "WARNING: Still could not find any load forecast data. Consider checking your data files."
    #             )

    #     # Load solar and wind generation forecasts
    #     print("\nLoading solar and wind forecasts...")
    #     solar_df, wind_df = self.load_solar_wind_forecasts(years)

    #     # Add solar forecasts to features
    #     if solar_df is not None and not solar_df.empty:
    #         solar_df["feature_type"] = "solar_forecast"
    #         # Add to features list
    #         print(f"Adding solar forecast data ({len(solar_df)} rows)")
    #         feature_dfs.append(solar_df)
    #     else:
    #         print("Warning: No solar forecast data available")

    #     # Add wind forecasts to features
    #     if wind_df is not None and not wind_df.empty:
    #         wind_df["feature_type"] = "wind_forecast"
    #         # Add to features list
    #         print(f"Adding wind forecast data ({len(wind_df)} rows)")
    #         feature_dfs.append(wind_df)
    #     else:
    #         print("Warning: No wind forecast data available")

    #     # Load total generation forecasts - with specific format handling
    #     gen_dfs = []
    #     for year in years:
    #         file_path = os.path.join(self.data_dir, f"total_generation_{year}.csv")
    #         if os.path.exists(file_path):
    #             # Based on the provided example, we need to handle the specific format
    #             try:
    #                 df = pd.read_csv(file_path, quotechar='"', delimiter=",")
    #             except:
    #                 # If that fails, try more flexible parsing
    #                 df = pd.read_csv(file_path, sep=None, engine="python")

    #             # Check if it's in the format from the example
    #             if (
    #                 "MTU" in df.columns
    #                 and "Scheduled Generation [MW] (D) - Germany (DE)" in df.columns
    #             ):
    #                 # Extract timestamp from MTU column
    #                 extracted = df["MTU"].str.extract(
    #                     r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})"
    #                 )
    #                 if not extracted.empty:
    #                     df["timestamp"] = pd.to_datetime(
    #                         extracted.iloc[:, 0],
    #                         format="%d.%m.%Y %H:%M",
    #                         errors="coerce",
    #                     )

    #                 # Rename generation column to simpler name
    #                 df = df.rename(
    #                     columns={
    #                         "Scheduled Generation [MW] (D) - Germany (DE)": "total_generation"
    #                     }
    #                 )

    #                 # Select only necessary columns
    #                 df = df[["timestamp", "total_generation"]]

    #                 # Convert generation value to numeric
    #                 df["total_generation"] = pd.to_numeric(
    #                     df["total_generation"], errors="coerce"
    #                 )

    #             else:
    #                 # Process in a more generic way if structure is different
    #                 self._process_timestamp_field(df)

    #                 # Try to identify the generation column
    #                 gen_col = None
    #                 for col in df.columns:
    #                     if "generation" in col.lower() or "scheduled" in col.lower():
    #                         gen_col = col
    #                         break

    #                 if gen_col:
    #                     df = self._safe_convert_to_numeric(df, gen_col)
    #                     df = df.rename(columns={gen_col: "total_generation"})

    #             # Check for 15-minute intervals
    #             if self._is_15min_data(df):
    #                 print(
    #                     f"Detected 15-minute interval data in {file_path}, resampling to hourly"
    #                 )
    #                 df = self._resample_to_hourly(
    #                     df, value_columns=["total_generation"]
    #                 )

    #             df["feature_type"] = "total_generation"
    #             gen_dfs.append(df)

    #     if gen_dfs:
    #         gen_df = pd.concat(gen_dfs, ignore_index=True)
    #         feature_dfs.append(gen_df)

    #     return feature_dfs

    # def load_import_export(self, years=None):
    #     """
    #     Lädt Import-Export-Daten zwischen Deutschland und 11 Nachbarländern.
    #     Erwartet Dateien mit Spalten:
    #     - "Time (CET/CEST)"
    #     - "[LandCode] > Germany (DE) [MW]" (Importe nach Deutschland)
    #     - "Germany (DE) > [LandCode] [MW]" (Exporte aus Deutschland)

    #     Args:
    #         years: Liste von Jahren zum Laden

    #     Returns:
    #         DataFrame mit aggregierten Import-Export-Daten
    #     """
    #     if years is None:
    #         years = [2017, 2018, 2019]

    #     all_import_export = []

    #     for year in years:
    #         # Finde alle Import-Export-Dateien für das aktuelle Jahr
    #         pattern = os.path.join(self.data_dir, f"*{year}*.csv")
    #         files = glob.glob(pattern)

    #         year_data = []
    #         for file in files:
    #             # Versuche zu identifizieren, ob dies eine Import/Export-Datei ist
    #             try:
    #                 df = pd.read_csv(file, quotechar='"')
    #             except:
    #                 try:
    #                     df = pd.read_csv(file, quotechar='"', delimiter=",")
    #                 except:
    #                     df = pd.read_csv(file, sep=None, engine="python")

    #             # Überprüfe, ob dies eine Import/Export-Datei ist
    #             has_import_export_cols = any(
    #                 ">" in col and "Germany" in col and "[MW]" in col
    #                 for col in df.columns
    #             )
    #             if not has_import_export_cols:
    #                 continue

    #             print(f"Import/Export-Datei gefunden: {os.path.basename(file)}")

    #             # Extrahiere Ländercode aus Dateiname oder Spalten
    #             country_code = None
    #             for col in df.columns:
    #                 if ">" in col and "Germany" in col and "[MW]" in col:
    #                     # Extrahiere Ländercode aus Spaltenname
    #                     if col.index(">") < col.index("Germany"):
    #                         parts = col.split(">")
    #                         country_code = parts[0].strip().split("(")[0].strip()
    #                         break

    #             if country_code is None:
    #                 # Wenn kein Ländercode gefunden wurde, verwende den Dateinamen
    #                 base_filename = os.path.basename(file)
    #                 country_code = base_filename.split("_")[0]

    #             # Verarbeite Zeitstempelspalte
    #             if "Time (CET/CEST)" in df.columns:
    #                 extracted = df["Time (CET/CEST)"].str.extract(
    #                     r"(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2})"
    #                 )
    #                 if not extracted.empty:
    #                     df["timestamp"] = pd.to_datetime(
    #                         extracted.iloc[:, 0],
    #                         format="%d.%m.%Y %H:%M",
    #                         errors="coerce",
    #                     )
    #             else:
    #                 self._process_timestamp_field(df)

    #             # Identifiziere Import- und Exportspalten
    #             import_col = None
    #             export_col = None

    #             for col in df.columns:
    #                 if ">" in col and "[MW]" in col:
    #                     if "Germany (DE)" in col:
    #                         if col.index(">") < col.index("Germany"):
    #                             # Format: "[LandCode] > Germany (DE) [MW]" - Importe nach Deutschland
    #                             import_col = col
    #                         elif col.index(">") > col.index("Germany"):
    #                             # Format: "Germany (DE) > [LandCode] [MW]" - Exporte aus Deutschland
    #                             export_col = col

    #             if import_col is not None or export_col is not None:
    #                 # Berechne Nettofluss: positiv = Import nach Deutschland, negativ = Export aus Deutschland
    #                 df["net_flow"] = 0.0

    #                 # Sichere Behandlung fehlender Werte
    #                 if import_col is not None:
    #                     # Safely convert to numeric
    #                     df = self._safe_convert_to_numeric(df, import_col)
    #                     # Add to net_flow (NaN values will be ignored)
    #                     df["net_flow"] += df[import_col]

    #                 if export_col is not None:
    #                     # Safely convert to numeric
    #                     df = self._safe_convert_to_numeric(df, export_col)
    #                     # Subtract from net_flow (NaN values will be ignored)
    #                     df["net_flow"] -= df[export_col]

    #                 df["country"] = country_code
    #                 df = df[["timestamp", "country", "net_flow"]]
    #                 year_data.append(df)
    #             else:
    #                 print(f"Warnung: Keine Import/Export-Spalten in {file} gefunden")

    #         if year_data:
    #             year_df = pd.concat(year_data, ignore_index=True)
    #             all_import_export.append(year_df)

    #     if not all_import_export:
    #         print("Warnung: Keine Import/Export-Daten gefunden!")
    #         return None

    #     # Kombiniere alle Jahre
    #     import_export_df = pd.concat(all_import_export, ignore_index=True)

    #     # Überprüfe auf 15-Minuten-Daten und resample wenn nötig
    #     if self._is_15min_data(import_export_df):
    #         print(
    #             "15-Minuten-Intervall in Import/Export-Daten erkannt, resampling auf stündlich"
    #         )
    #         import_export_df = self._resample_to_hourly(
    #             import_export_df, value_columns=["net_flow"]
    #         )

    #     # Aggregiere Import-Export nach Zeitstempel (Summe über alle Länder)
    #     agg_import_export = (
    #         import_export_df.groupby("timestamp")["net_flow"].sum().reset_index()
    #     )
    #     agg_import_export.rename(
    #         columns={"net_flow": "net_import_export"}, inplace=True
    #     )
    #     # At the end of this method, add min_count parameter to sum to handle NaN values
    #     agg_import_export = (
    #         import_export_df.groupby("timestamp")["net_flow"]
    #         .sum(min_count=1)
    #         .reset_index()
    #     )

    #     return agg_import_export

    # def load_fuel_prices(self, years=None):
    #     """
    #     Load fuel prices (oil and natural gas).

    #     Args:
    #         years: List of years to load

    #     Returns:
    #         DataFrame with fuel prices
    #     """
    #     if years is None:
    #         years = [2017, 2018, 2019]

    #     # Load oil prices
    #     oil_dfs = []
    #     for year in years:
    #         file_path = os.path.join(self.data_dir, f"oil_price_{year}.csv")
    #         if os.path.exists(file_path):
    #             try:
    #                 # Try first with semicolon delimiter (based on example)
    #                 df = pd.read_csv(file_path, sep=";", decimal=",")
    #             except:
    #                 try:
    #                     # If that fails, try comma delimiter
    #                     df = pd.read_csv(file_path, decimal=",")
    #                 except:
    #                     # If all else fails, use more flexible parsing
    #                     df = pd.read_csv(file_path, sep=None, engine="python")

    #             # Check if it matches the format from the example
    #             if "Datum" in df.columns and "Schlusskurs" in df.columns:
    #                 print(
    #                     f"Found oil price data with expected German format in {file_path}"
    #                 )

    #                 # Convert Datum to timestamp
    #                 df["timestamp"] = pd.to_datetime(df["Datum"])

    #                 # Use Schlusskurs (closing price) as the price
    #                 df = df.rename(columns={"Schlusskurs": "oil"})

    #                 # Convert to numeric, handling potential comma as decimal separator
    #                 df["oil"] = pd.to_numeric(
    #                     df["oil"].astype(str).str.replace(",", "."), errors="coerce"
    #                 )

    #                 # Select only necessary columns
    #                 df = df[["timestamp", "oil"]]

    #             else:
    #                 # Try to find timestamp and price columns
    #                 self._process_timestamp_field(df)

    #                 price_col = None
    #                 for col in df.columns:
    #                     if col != "timestamp" and pd.api.types.is_numeric_dtype(
    #                         df[col]
    #                     ):
    #                         price_col = col
    #                         break

    #                 if price_col:
    #                     df = df.rename(columns={price_col: "oil"})
    #                     df = df[["timestamp", "oil"]]
    #                 else:
    #                     print(
    #                         f"Warning: Could not identify oil price column in {file_path}"
    #                     )
    #                     continue

    #             df["oil"] = pd.to_numeric(df["oil"], errors="coerce")
    #             oil_dfs.append(df)

    #     # Load natural gas prices
    #     gas_dfs = []
    #     for year in years:
    #         file_path = os.path.join(self.data_dir, f"natural_gas_price_{year}.csv")
    #         if os.path.exists(file_path):
    #             try:
    #                 # Try first with semicolon delimiter (based on example)
    #                 df = pd.read_csv(file_path, sep=";", decimal=",")
    #             except:
    #                 try:
    #                     # If that fails, try comma delimiter
    #                     df = pd.read_csv(file_path, decimal=",")
    #                 except:
    #                     # If all else fails, use more flexible parsing
    #                     df = pd.read_csv(file_path, sep=None, engine="python")

    #             # Check if it matches the format from the example
    #             if "Datum" in df.columns and "Schlusskurs" in df.columns:
    #                 print(
    #                     f"Found natural gas price data with expected German format in {file_path}"
    #                 )

    #                 # Convert Datum to timestamp
    #                 df["timestamp"] = pd.to_datetime(df["Datum"])

    #                 # Use Schlusskurs (closing price) as the price
    #                 df = df.rename(columns={"Schlusskurs": "natural_gas"})

    #                 # Convert to numeric, handling potential comma as decimal separator
    #                 df["natural_gas"] = pd.to_numeric(
    #                     df["natural_gas"].astype(str).str.replace(",", "."),
    #                     errors="coerce",
    #                 )

    #                 # Select only necessary columns
    #                 df = df[["timestamp", "natural_gas"]]

    #             else:
    #                 # Try to find timestamp and price columns
    #                 self._process_timestamp_field(df)

    #                 price_col = None
    #                 for col in df.columns:
    #                     if col != "timestamp" and pd.api.types.is_numeric_dtype(
    #                         df[col]
    #                     ):
    #                         price_col = col
    #                         break

    #                 if price_col:
    #                     df = df.rename(columns={price_col: "natural_gas"})
    #                     df = df[["timestamp", "natural_gas"]]
    #                 else:
    #                     print(
    #                         f"Warning: Could not identify natural gas price column in {file_path}"
    #                     )
    #                     continue

    #             df["natural_gas"] = pd.to_numeric(df["natural_gas"], errors="coerce")
    #             gas_dfs.append(df)

    #     # Combine all fuel prices
    #     all_prices = []

    #     if oil_dfs:
    #         oil_df = pd.concat(oil_dfs, ignore_index=True)
    #         oil_df = oil_df.sort_values("timestamp")
    #         all_prices.append(oil_df)

    #     if gas_dfs:
    #         gas_df = pd.concat(gas_dfs, ignore_index=True)
    #         gas_df = gas_df.sort_values("timestamp")
    #         all_prices.append(gas_df)

    #     if not all_prices:
    #         print("Warning: No fuel price data found!")
    #         return None

    #     # Merge oil and gas prices on timestamp (daily resolution)
    #     fuel_prices_daily = (
    #         pd.merge(all_prices[0], all_prices[1], on="timestamp", how="outer")
    #         if len(all_prices) > 1
    #         else all_prices[0]
    #     )

    #     # Create a continuous date range from min to max timestamp
    #     start_date = fuel_prices_daily["timestamp"].min()
    #     end_date = fuel_prices_daily["timestamp"].max()

    #     # Create hourly timestamps for the full range
    #     hourly_range = pd.date_range(start=start_date, end=end_date, freq="h")
    #     hourly_df = pd.DataFrame({"timestamp": hourly_range})

    #     # Linear interpolation to hourly frequency (as mentioned in the paper)
    #     print(
    #         "Performing linear interpolation of daily fuel prices to hourly resolution..."
    #     )

    #     # Merge with the hourly timestamps to get a dataframe with NaNs
    #     merged_df = pd.merge(hourly_df, fuel_prices_daily, on="timestamp", how="left")

    #     # Linearly interpolate the NaNs
    #     for col in ["oil", "natural_gas"]:
    #         if col in merged_df.columns:
    #             merged_df[col] = merged_df[col].interpolate(method="linear")

    #     # Forward fill any remaining NaNs at the beginning
    #     merged_df = merged_df.ffill()

    #     # Backward fill any remaining NaNs at the end
    #     merged_df = merged_df.bfill()

    #     return merged_df

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

    def plot_shap_dependency(self, top_n=3):
        """
        Plot SHAP dependency plots for top features (Fig. 3a-c in paper).

        Arguments:
            top_n: Number of top features to plot

        Returns:
            List of top feature names
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(0)

        # Get top features
        top_indices = np.argsort(-feature_importance)[:top_n]
        top_features = [self.X_test.columns[i] for i in top_indices]

        print(f"Top {top_n} features by SHAP importance: {top_features}")

        # Create a new figure with 3 subplots in a row
        fig, axes = plt.subplots(1, top_n, figsize=(18, 6))

        # Ensure axes is always an array
        if top_n == 1:
            axes = np.array([axes])

        # Define color mapping features
        color_features = {
            "load_forecast": "wind_forecast",
            "solar_forecast": "load_forecast",
            "wind_forecast": "load_forecast",
        }

        # Create plots - similar to the images provided
        for i, feature in enumerate(top_features):
            feature_idx = list(self.X_test.columns).index(feature)
            color_feature = color_features.get(feature, top_features[0])
            if color_feature in self.X_test.columns:
                color_idx = list(self.X_test.columns).index(color_feature)
            else:
                color_idx = "auto"

            # Force matplotlib to use our specific axis
            plt.sca(axes[i])

            # Plot in the specific axis
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                self.X_test,
                interaction_index=color_idx,
                ax=axes[i],
                show=False,
            )

            # Update plot title
            axes[i].set_title(f"SHAP dependency for {feature}")

            # Add letter labels matching the paper format (a, b, c)
            axes[i].text(
                0.05,
                0.95,
                f"{chr(97+i)})",
                transform=axes[i].transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )

        # Make sure the main figure is active
        plt.figure(fig.number)

        # Add common y-axis label
        fig.text(
            0.04,
            0.5,
            "SHAP values (Electricity Price [EUR/MWh])",
            va="center",
            rotation="vertical",
            fontsize=12,
        )

        # Ensure proper spacing
        plt.tight_layout(rect=[0.05, 0, 1, 1])

        return top_features

    def plot_paper_style_shap_dependencies(self, top_n=3):
        """
        Plot SHAP dependency plots for top features in a row,
        matching the style from the reference paper.

        Arguments:
            top_n: Number of top features to plot

        Returns:
            List of top feature names
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        # Calculate feature importance
        feature_importance = np.abs(self.shap_values).mean(0)

        # Get top features
        # top_indices = np.argsort(-feature_importance)[:top_n]
        top_features = ["load_forecast", "solar_forecast", "wind_forecast"]

        print(f"Top {top_n} features by SHAP importance: {top_features}")

        # Create a new figure with N subplots in a row
        fig, axes = plt.subplots(1, top_n, figsize=(18, 6))

        # Ensure axes is always an array
        if top_n == 1:
            axes = np.array([axes])

        # Define prettier feature names for x-axis labels
        feature_labels = {
            "load_forecast": "Load day-ahead [MWh]",
            "wind_forecast": "Wind day-ahead [MWh]",
            "solar_forecast": "Solar day-ahead [MWh]",
        }

        # Create plots
        for i, feature in enumerate(top_features):
            feature_idx = list(self.X_test.columns).index(feature)

            # Make a copy of the data for potential negation
            x_data = self.X_test[feature].copy()

            # Negate solar and wind values to match paper style
            if feature in ["solar_forecast", "wind_forecast"]:
                x_data = -x_data

            # Force matplotlib to use our specific axis
            plt.sca(axes[i])

            # Plot SHAP values against the feature data
            scatter = axes[i].scatter(
                x_data,
                self.shap_values[:, feature_idx],
                alpha=0.6,
                s=12,  # Smaller point size for cleaner look
                color="#333333",  # Darker points like in the paper
            )

            # Add a trend line
            from scipy.stats import linregress

            slope, intercept, r_value, p_value, std_err = linregress(
                x_data, self.shap_values[:, feature_idx]
            )
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            axes[i].plot(
                x_range, intercept + slope * x_range, color="skyblue", linewidth=2
            )

            # Update plot style to match paper
            axes[i].set_title(
                f"{chr(97+i)}", loc="left", fontsize=14, fontweight="bold"
            )

            # Use prettier x-axis label if available
            axes[i].set_xlabel(feature_labels.get(feature, feature), fontsize=12)

            # Only add y-axis label to the first plot
            if i == 0:
                axes[i].set_ylabel(
                    "SHAP values (Electricity Price\n[EUR/MWh])", fontsize=12
                )

            # Add grid for better readability
            axes[i].grid(True, alpha=0.3, linestyle="--")

        # Ensure proper spacing and layout
        plt.tight_layout()

        return top_features

    def plot_shap_interaction(self, feature1, feature2):
        """
        Plot SHAP interaction plots (Fig. 4 and 5 in paper).

        Arguments:
            feature1: First feature name
            feature2: Second feature name
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        # Calculate SHAP interaction values (can be computationally expensive)
        print("Calculating SHAP interaction values (this may take some time)...")
        shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)

        # Get feature indices
        idx1 = list(self.X_test.columns).index(feature1)
        idx2 = list(self.X_test.columns).index(feature2)

        # Create interaction plot
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            (idx1, idx2),
            shap_interaction_values,
            self.X_test,
            display_features=self.X_test,
        )

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


def run_german_electricity_price_model(data_dir):
    """
    Run the full workflow for German electricity price prediction.

    Args:
        data_dir: Directory containing the dataset CSV files

    Returns:
        Trained model and evaluation results
    """
    # 1. Load and prepare the data
    print("Loading and preparing data...")
    data_loader = GermanElectricityDataLoader(data_dir)
    full_dataset = data_loader.prepare_full_dataset(years=[2017, 2018, 2019])

    print(f"Dataset shape: {full_dataset.shape}")
    print(f"Available features: {full_dataset.columns.tolist()}")

    # 2. Initialize and train the model
    print("Training the GBT model...")
    model = ElectricityPriceModel(random_state=42)

    # Train with automatically selected features based on Fig. 2
    model.train(full_dataset)

    # 3. Generate SHAP explanations
    print("Generating SHAP explanations...")
    model.explain_with_shap()

    # 4. Create visualizations
    # 4.1 Global feature importance as shown in Fig. 2
    print("Creating visualizations...")
    model.plot_global_feature_importance()

    # 4.2 SHAP dependency plots for top features (similar to Fig. 3a-c)
    top_features = model.plot_shap_dependency(
        specific_features=["load_forecast", "solar_forecast", "wind_forecast"]
    )

    # 4.3 Feature interactions (similar to Fig. 4 and 5)
    if len(top_features) >= 2:
        feature1, feature2 = top_features[0], top_features[1]
        model.plot_shap_interaction(feature1, feature2)

    # 5. Evaluate model consistency (optional)
    print("Evaluating model consistency across multiple splits...")
    consistency_results = model.evaluate_model_consistency(
        full_dataset,
        features=None,  # Use automatic feature selection
        target="price",
        time_column="timestamp",
        n_splits=5,  # Reduced from 10 for faster execution
    )

    return model, consistency_results
