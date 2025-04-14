"""
Enhanced version of the GermanElectricityDataLoader class that reads Excel files
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
from timestamp_parser import parse_timestamps


class ExcelElectricityDataLoader:
    """
    Data loader for German electricity price prediction datasets.
    Handles loading and preprocessing of Excel files.
    """

    def __init__(self, data_dir):
        """Initialize the data loader with the directory containing Excel files"""
        self.data_dir = data_dir

    def _read_excel_file(self, file_path):
        """
        Read an Excel file and standardize its timestamp handling using our enhanced parser.

        Args:
            file_path: Path to the Excel file

        Returns:
            DataFrame with standardized format
        """
        try:
            print(f"Reading Excel file: {file_path}")
            # Read the Excel file - use engine='openpyxl' for better handling of various Excel formats
            df = pd.read_excel(file_path, engine='openpyxl')

            # Debug output
            print(f"  Columns found: {df.columns.tolist()}")
            print(f"  Row count: {len(df)}")

            # Validate that we have the expected timestamp column
            if 'timestamp' in df.columns:
                print(f"  Found timestamp column: 'timestamp'")

                # Use our enhanced timestamp parser
                df = parse_timestamps(df, timestamp_column='timestamp')

                # Check if timestamps were successfully parsed
                valid_count = df['timestamp'].notna().sum()
                print(
                    f"  Final timestamp status: {valid_count} valid timestamps out of {len(df)} rows")

                if valid_count > 0 and len(df) > 0:
                    print(f"  Sample timestamp: {df['timestamp'].iloc[0]}")
            else:
                print(f"  WARNING: No 'timestamp' column found in {file_path}")
                print(f"  Available columns: {df.columns.tolist()}")

                # Try to identify a timestamp column
                timestamp_candidates = [
                    col for col in df.columns if any(keyword in col.lower()
                                                     for keyword in ['time', 'date', 'timestamp', 'mtu', 'zeit'])
                ]

                if timestamp_candidates:
                    timestamp_col = timestamp_candidates[0]
                    print(
                        f"  Using alternative timestamp column: '{timestamp_col}'")

                    # Rename the column to 'timestamp'
                    df = df.rename(columns={timestamp_col: 'timestamp'})

                    # Parse the timestamps
                    df = parse_timestamps(df, timestamp_column='timestamp')
                else:
                    print(
                        f"  No suitable timestamp column found in {file_path}")
                    return None

            return df

        except Exception as e:
            print(f"Error reading Excel file {file_path}: {str(e)}")
            return None

    def _safe_convert_to_numeric(self, df, column):
        """
        Safely convert a column to numeric by replacing common missing value indicators
        and using pd.to_numeric with errors='coerce'.
        """
        if column in df.columns:
            # Replace common missing value indicators with NaN
            df[column] = df[column].replace(
                ['n/e', 'N/A', 'n.e.', 'n/a', '-'], np.nan)
            # Convert to numeric, coercing errors to NaN
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def load_electricity_prices(self, years=None):
        """
        Load hourly day-ahead electricity prices from Excel files.

        Args:
            years: List of years to load (e.g., [2017, 2018, 2019])
                If None, load all available years

        Returns:
            DataFrame with electricity prices
        """
        if years is None:
            years = [2017, 2018, 2019]

        price_dfs = []

        for year in years:
            # Try both naming conventions
            possible_files = [
                f"electricity_prices_{year}.xlsx",
                f"electricity_{year}.xlsx",
                f"prices_{year}.xlsx",
                f"day_ahead_prices_{year}.xlsx"
            ]

            file_found = False
            for filename in possible_files:
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    file_found = True
                    df = self._read_excel_file(file_path)

                    if df is not None:
                        # Identify price column
                        price_col = None

                        # Look for price-related column names
                        price_candidates = [
                            col for col in df.columns if any(keyword in col.lower()
                                                             for keyword in ['price', 'preis', 'eur', '€', 'euro', 'day-ahead'])
                        ]

                        if price_candidates:
                            price_col = price_candidates[0]
                            print(f"  Found price column: '{price_col}'")
                        else:
                            # If no obvious price column, use first numeric column that's not timestamp
                            for col in df.columns:
                                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                                    price_col = col
                                    print(
                                        f"  Using first numeric column '{price_col}' as price data")
                                    break

                        if price_col:
                            # Safely convert to numeric and rename
                            df = self._safe_convert_to_numeric(df, price_col)
                            df = df.rename(columns={price_col: 'price'})

                            # Select only necessary columns
                            df = df[['timestamp', 'price']]
                            price_dfs.append(df)
                        else:
                            print(
                                f"  Warning: Could not identify price column in {file_path}")

                    # Break once we've found and processed a file
                    break

            if not file_found:
                print(
                    f"Warning: No electricity price Excel file found for {year}")

        if not price_dfs:
            raise ValueError("No electricity price data found or processed!")

        # Combine all years
        prices_df = pd.concat(price_dfs, ignore_index=True)

        # Sort by timestamp
        prices_df = prices_df.sort_values('timestamp')

        return prices_df

    def load_solar_wind_forecasts(self, years=None):
        """
        Load solar and wind generation forecasts from Excel files.

        Args:
            years: List of years to load

        Returns:
            Tuple of DataFrames (solar_df, wind_df)
        """
        if years is None:
            years = [2017, 2018, 2019]

        solar_dfs = []
        wind_dfs = []

        for year in years:
            # Try various possible filenames
            possible_files = [
                f"solar_wind_forecast_{year}.xlsx",
                f"wind_solar_forecast_{year}.xlsx",
                f"renewable_forecast_{year}.xlsx"
            ]

            file_found = False
            for filename in possible_files:
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    file_found = True
                    print(
                        f"Found combined solar and wind forecast file: {file_path}")

                    df = self._read_excel_file(file_path)

                    if df is None or 'timestamp' not in df.columns:
                        print(
                            f"  Skipping file {file_path}, no valid timestamp column")
                        continue

                    # Identify solar day ahead column
                    solar_col = None
                    for col in df.columns:
                        if 'solar' in col.lower() and ('day ahead' in col.lower() or 'forecast' in col.lower()):
                            solar_col = col
                            break

                    # Identify wind onshore and offshore day ahead columns
                    wind_onshore_col = None
                    wind_offshore_col = None

                    for col in df.columns:
                        if 'wind onshore' in col.lower() and ('day ahead' in col.lower() or 'forecast' in col.lower()):
                            wind_onshore_col = col
                        elif 'wind offshore' in col.lower() and ('day ahead' in col.lower() or 'forecast' in col.lower()):
                            wind_offshore_col = col

                    # If not found specific onshore/offshore, look for general wind
                    if wind_onshore_col is None and wind_offshore_col is None:
                        for col in df.columns:
                            if 'wind' in col.lower() and ('day ahead' in col.lower() or 'forecast' in col.lower()):
                                wind_onshore_col = col
                                break

                    # Process solar data
                    if solar_col is not None:
                        print(f"  Found solar forecast column: '{solar_col}'")
                        # Safely convert to numeric
                        df = self._safe_convert_to_numeric(df, solar_col)

                        solar_data = df[['timestamp', solar_col]].copy()
                        solar_data = solar_data.rename(
                            columns={solar_col: 'solar_forecast'})

                        # Check for 15-minute intervals and resample if needed
                        if self._is_15min_data(solar_data):
                            print(
                                f"  Detected 15-minute interval solar data, resampling to hourly")
                            solar_data = self._resample_to_hourly(
                                solar_data, value_columns=['solar_forecast'])

                        solar_dfs.append(solar_data)

                    # Process wind data
                    if wind_onshore_col is not None or wind_offshore_col is not None:
                        print(
                            f"  Found wind forecast columns: Onshore={wind_onshore_col}, Offshore={wind_offshore_col}")
                        wind_data = df[['timestamp']].copy()

                        # First, safely convert the columns to numeric values
                        if wind_onshore_col is not None:
                            df = self._safe_convert_to_numeric(
                                df, wind_onshore_col)

                        if wind_offshore_col is not None:
                            df = self._safe_convert_to_numeric(
                                df, wind_offshore_col)

                        # Now combine the data based on what's available
                        if wind_onshore_col is not None and wind_offshore_col is not None:
                            # Sum onshore and offshore wind
                            wind_data['wind_forecast'] = df[wind_onshore_col].add(
                                df[wind_offshore_col], fill_value=0)
                        elif wind_onshore_col is not None:
                            wind_data['wind_forecast'] = df[wind_onshore_col]
                        elif wind_offshore_col is not None:
                            wind_data['wind_forecast'] = df[wind_offshore_col]

                        # Check for 15-minute intervals and resample if needed
                        if self._is_15min_data(wind_data):
                            print(
                                f"  Detected 15-minute interval wind data, resampling to hourly")
                            wind_data = self._resample_to_hourly(
                                wind_data, value_columns=['wind_forecast'])

                        wind_dfs.append(wind_data)

                    # Break out of the filename loop if we found a file
                    break

            if not file_found:
                print(f"No combined solar/wind Excel data found for {year}")

        # Combine all years
        solar_df = pd.concat(solar_dfs, ignore_index=True) if len(
            solar_dfs) > 0 else None
        wind_df = pd.concat(wind_dfs, ignore_index=True) if len(
            wind_dfs) > 0 else None

        # Add debugging info
        if solar_df is not None:
            print(
                f"Solar forecast data: {len(solar_df)} rows, range: {solar_df['timestamp'].min()} to {solar_df['timestamp'].max()}")
        else:
            print("No solar forecast data found!")

        if wind_df is not None:
            print(
                f"Wind forecast data: {len(wind_df)} rows, range: {wind_df['timestamp'].min()} to {wind_df['timestamp'].max()}")
        else:
            print("No wind forecast data found!")

        return solar_df, wind_df

    def load_power_system_features(self, years=None):
        """
        Load power system features from Excel files:
        - Day-ahead forecasts of load
        - Day-ahead total generation

        Args:
            years: List of years to load

        Returns:
            DataFrame with power system features
        """
        if years is None:
            years = [2017, 2018, 2019]

        feature_dfs = []

        # Load load forecasts
        load_dfs = []
        for year in years:
            file_path = os.path.join(
                self.data_dir, f"load_forecast_{year}.xlsx")
            if os.path.exists(file_path):
                # print(f"Loading load forecast file: {file_path}")
                df = self._read_excel_file(file_path)

                if df is None or 'timestamp' not in df.columns:
                    print(
                        f"  Skipping file {file_path}, no valid timestamp column")
                    continue

                # Identify the forecast column
                forecast_col = None
                for col in df.columns:
                    if "forecast" in col.lower() or "load forecast" in col.lower():
                        forecast_col = col
                        break

                # If no forecast column found, look for a column with 'load' in the name
                if forecast_col is None:
                    for col in df.columns:
                        if "load" in col.lower() and "actual" not in col.lower():
                            forecast_col = col
                            break

                # If still no column found, use the first numeric column that's not timestamp
                if forecast_col is None:
                    for col in df.columns:
                        if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                            forecast_col = col
                            break

                if forecast_col:
                    print(f"  Using column '{forecast_col}' for load forecast")

                    # Print sample values from this column
                    if len(df) > 0:
                        print(
                            f"  Sample values: {df[forecast_col].head(3).tolist()}")

                    # Ensure safe conversion to numeric
                    df = self._safe_convert_to_numeric(df, forecast_col)

                    # Check if all values became NaN after conversion
                    if df[forecast_col].isna().all():
                        print(
                            f"  WARNING: All values in '{forecast_col}' became NaN after conversion")

                    # Rename the column
                    df = df.rename(columns={forecast_col: "load_forecast"})

                    # Check for 15-minute intervals and resample if needed
                    if self._is_15min_data(df):
                        print(
                            f"  Detected 15-minute interval load data, resampling to hourly")
                        df = self._resample_to_hourly(
                            df, value_columns=["load_forecast"])

                    df["feature_type"] = "load_forecast"
                    df = df[["timestamp", "load_forecast", "feature_type"]]
                    load_dfs.append(df)
                else:
                    print(
                        f"  Warning: Could not identify load forecast column in {file_path}")

        if load_dfs:
            load_df = pd.concat(load_dfs, ignore_index=True)
            feature_dfs.append(load_df)
            print(f"Combined load forecast data: {len(load_df)} rows")

        # Load solar and wind generation forecasts
        # print("\nLoading solar and wind forecasts...")
        solar_df, wind_df = self.load_solar_wind_forecasts(years)

        # Add solar forecasts to features
        if solar_df is not None and not solar_df.empty:
            solar_df["feature_type"] = "solar_forecast"
            # print(f"Adding solar forecast data ({len(solar_df)} rows)")
            feature_dfs.append(solar_df)
        else:
            print("Warning: No solar forecast data available")

        # Add wind forecasts to features
        if wind_df is not None and not wind_df.empty:
            wind_df["feature_type"] = "wind_forecast"
            # print(f"Adding wind forecast data ({len(wind_df)} rows)")
            feature_dfs.append(wind_df)
        else:
            print("Warning: No wind forecast data available")

        # Load total generation forecasts
        gen_dfs = []
        for year in years:
            file_path = os.path.join(
                self.data_dir, f"total_generation_{year}.xlsx")
            if os.path.exists(file_path):
                # print(f"Loading total generation file: {file_path}")
                df = self._read_excel_file(file_path)

                if df is None or 'timestamp' not in df.columns:
                    print(
                        f"  Skipping file {file_path}, no valid timestamp column")
                    continue

                # Try to identify the generation column
                gen_col = None
                for col in df.columns:
                    if "generation" in col.lower() or "scheduled" in col.lower():
                        gen_col = col
                        break

                if gen_col:
                    print(f"  Using column '{gen_col}' for total generation")
                    df = self._safe_convert_to_numeric(df, gen_col)
                    df = df.rename(columns={gen_col: "total_generation"})

                    # Check for 15-minute intervals
                    if self._is_15min_data(df):
                        print(
                            f"  Detected 15-minute interval data, resampling to hourly")
                        df = self._resample_to_hourly(
                            df, value_columns=["total_generation"])

                    df["feature_type"] = "total_generation"
                    gen_dfs.append(df)
                else:
                    print(
                        f"  Warning: Could not identify generation column in {file_path}")

        if gen_dfs:
            gen_df = pd.concat(gen_dfs, ignore_index=True)
            feature_dfs.append(gen_df)
            # print(f"Combined total generation data: {len(gen_df)} rows")

        return feature_dfs

    def load_import_export(self, years=None):
        """
        Load Import-Export data between Germany and neighboring countries from Excel files.

        Args:
            years: List of years to load

        Returns:
            DataFrame with aggregated import-export data
        """
        if years is None:
            years = [2017, 2018, 2019]

        all_import_export = []

        for year in years:
            # Find all Import-Export Excel files for this year
            pattern = os.path.join(self.data_dir, f"*{year}*.xlsx")
            files = glob.glob(pattern)

            year_data = []
            for file in files:
                # Skip files that are not import/export related
                base_filename = os.path.basename(file).lower()
                if not (('import' in base_filename) or ('export' in base_filename) or
                        any(cc in base_filename for cc in ['at_', 'ch_', 'fr_', 'nl_', 'pl_', 'cz_', 'dk_'])):
                    continue

                # print(f"Examining potential import/export file: {os.path.basename(file)}")
                df = self._read_excel_file(file)

                if df is None or 'timestamp' not in df.columns:
                    print(f"  Skipping file {file}, no valid timestamp column")
                    continue

                # Try to extract country code from filename
                base_filename = os.path.basename(file)
                country_code = base_filename.split('_')[0].upper()

                # Look for import/export columns
                import_col = None
                export_col = None

                # Check columns for import/export patterns
                for col in df.columns:
                    if 'import' in col.lower() or ('>' in col and 'germany' in col.lower()):
                        import_col = col
                    elif 'export' in col.lower() or ('germany' in col.lower() and '>' in col):
                        export_col = col

                if import_col is not None or export_col is not None:
                    # print(f"  Found import/export columns for {country_code}: Import={import_col}, Export={export_col}")

                    # Calculate net flow (positive=import, negative=export)
                    df['net_flow'] = 0.0

                    if import_col is not None:
                        df = self._safe_convert_to_numeric(df, import_col)
                        df['net_flow'] += df[import_col]

                    if export_col is not None:
                        df = self._safe_convert_to_numeric(df, export_col)
                        df['net_flow'] -= df[export_col]

                    df['country'] = country_code
                    df = df[['timestamp', 'country', 'net_flow']]

                    # Check for 15-minute intervals
                    if self._is_15min_data(df):
                        print(
                            f"  Detected 15-minute interval data, resampling to hourly")
                        df = self._resample_to_hourly(
                            df, value_columns=['net_flow'])

                    year_data.append(df)
                else:
                    print(f"  No import/export columns found in {file}")

            if year_data:
                year_df = pd.concat(year_data, ignore_index=True)
                all_import_export.append(year_df)

        if not all_import_export:
            print("Warning: No import/export data found!")
            return None

        # Combine all years
        import_export_df = pd.concat(all_import_export, ignore_index=True)

        # Aggregate by timestamp (sum across all countries)
        agg_import_export = import_export_df.groupby(
            'timestamp')['net_flow'].sum(min_count=1).reset_index()
        agg_import_export.rename(
            columns={'net_flow': 'net_import_export'}, inplace=True)

        # print(f"Aggregated import/export data: {len(agg_import_export)} rows")

        return agg_import_export

    def load_fuel_prices(self, years=None):
        """
        Load fuel prices (oil and natural gas) with improved interpolation.

        Args:
            years: List of years to load

        Returns:
            DataFrame with fuel prices
        """
        if years is None:
            years = [2017, 2018, 2019]

        # Load oil prices
        oil_dfs = []
        for year in years:
            file_path = os.path.join(self.data_dir, f"oil_price_{year}.xlsx")
            if os.path.exists(file_path):
                # print(f"Loading oil price file: {file_path}")
                df = self._read_excel_file(file_path)

                if df is None or 'timestamp' not in df.columns:
                    print(
                        f"  Skipping file {file_path}, no valid timestamp column")
                    continue

                # Try to identify the price column
                price_col = None
                for col in df.columns:
                    if 'price' in col.lower() or 'schlusskurs' in col.lower() or 'oil' in col.lower():
                        price_col = col
                        break

                # If no price column found, try the first numeric column
                if price_col is None:
                    for col in df.columns:
                        if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                            price_col = col
                            break

                if price_col:
                    print(f"  Using column '{price_col}' for oil price")
                    df = self._safe_convert_to_numeric(df, price_col)
                    df = df.rename(columns={price_col: 'oil'})
                    df = df[['timestamp', 'oil']]

                    # Debug
                    # print(f"  Oil price data: {len(df)} rows, range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    # if len(df) > 0:
                    #     print(f"  Sample values: {df['oil'].head(3).tolist()}")

                    oil_dfs.append(df)
                else:
                    print(
                        f"  Warning: Could not identify oil price column in {file_path}")

        # Load natural gas prices
        gas_dfs = []
        for year in years:
            file_path = os.path.join(
                self.data_dir, f"natural_gas_price_{year}.xlsx")
            if os.path.exists(file_path):
                # print(f"Loading natural gas price file: {file_path}")
                df = self._read_excel_file(file_path)

                if df is None or 'timestamp' not in df.columns:
                    print(
                        f"  Skipping file {file_path}, no valid timestamp column")
                    continue

                # Try to identify the price column
                price_col = None
                for col in df.columns:
                    if 'price' in col.lower() or 'schlusskurs' in col.lower() or 'gas' in col.lower():
                        price_col = col
                        break

                # If no price column found, try the first numeric column
                if price_col is None:
                    for col in df.columns:
                        if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                            price_col = col
                            break

                if price_col:
                    print(
                        f"  Using column '{price_col}' for natural gas price")
                    df = self._safe_convert_to_numeric(df, price_col)
                    df = df.rename(columns={price_col: 'natural_gas'})
                    df = df[['timestamp', 'natural_gas']]

                    # Debug
                    print(
                        f"  Natural gas price data: {len(df)} rows, range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                    if len(df) > 0:
                        print(
                            f"  Sample values: {df['natural_gas'].head(3).tolist()}")

                    gas_dfs.append(df)
                else:
                    print(
                        f"  Warning: Could not identify gas price column in {file_path}")

        # Combine all fuel prices
        # print("\nProcessing fuel price data:")
        all_prices = []

        if oil_dfs:
            oil_df = pd.concat(oil_dfs, ignore_index=True)
            oil_df = oil_df.sort_values('timestamp')
            all_prices.append(oil_df)
            print(f"  Combined oil price data: {len(oil_df)} rows")
        else:
            print("  Warning: No oil price data found!")

        if gas_dfs:
            gas_df = pd.concat(gas_dfs, ignore_index=True)
            gas_df = gas_df.sort_values('timestamp')
            all_prices.append(gas_df)
            print(f"  Combined natural gas price data: {len(gas_df)} rows")
        else:
            print("  Warning: No natural gas price data found!")

        if not all_prices:
            print("  Warning: No fuel price data found!")
            return None

        # Merge oil and gas prices on timestamp (daily resolution)
        # print("  Merging oil and gas price data")
        fuel_prices_daily = pd.merge(all_prices[0], all_prices[1], on='timestamp', how='outer') if len(
            all_prices) > 1 else all_prices[0]

        # Sort by timestamp
        fuel_prices_daily = fuel_prices_daily.sort_values('timestamp')

        # Debug merged data
        # print(f"  Merged fuel prices: {len(fuel_prices_daily)} rows")
        # print(f"  Date range: {fuel_prices_daily['timestamp'].min()} to {fuel_prices_daily['timestamp'].max()}")

        # Before interpolation, check if all columns are numeric
        for col in fuel_prices_daily.columns:
            if col != 'timestamp':
                if not pd.api.types.is_numeric_dtype(fuel_prices_daily[col]):
                    print(
                        f"  Warning: Column '{col}' is not numeric. Converting...")
                    fuel_prices_daily[col] = pd.to_numeric(
                        fuel_prices_daily[col], errors='coerce')

        # Create a continuous date range from min to max timestamp
        start_date = fuel_prices_daily['timestamp'].min()
        end_date = fuel_prices_daily['timestamp'].max()

        # Extend the range to cover the full year if needed
        if years and len(years) == 1:
            year = years[0]
            min_date = pd.Timestamp(f"{year}-01-01")
            max_date = pd.Timestamp(f"{year}-12-31 23:59:59")

            start_date = min(
                start_date, min_date) if start_date is not None else min_date
            end_date = max(
                end_date, max_date) if end_date is not None else max_date

            print(
                f"  Extended date range to cover full year: {start_date} to {end_date}")

        # Create hourly timestamps for the full range
        # print("  Creating hourly timestamp grid")
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
        hourly_df = pd.DataFrame({'timestamp': hourly_range})

        # print(f"  Hourly grid: {len(hourly_df)} hours")

        # Linear interpolation to hourly frequency
        # print("  Performing linear interpolation of daily fuel prices to hourly resolution...")

        # Merge with the hourly timestamps to get a dataframe with NaNs
        merged_df = pd.merge(hourly_df, fuel_prices_daily,
                             on='timestamp', how='left')

        # Debug before interpolation
        # print(f"  Before interpolation:")
        for col in merged_df.columns:
            if col != 'timestamp':
                missing = merged_df[col].isna().sum()
                print(
                    f"    {col}: {merged_df[col].notna().sum()} non-null values, {missing} missing values")

        # Interpolate the NaNs
        for col in merged_df.columns:
            if col != 'timestamp':
                # Interpolate linearly
                merged_df[col] = merged_df[col].interpolate(
                    method='linear', limit_direction='both')

                # If any values still missing at the beginning/end, use forward/backward fill
                if merged_df[col].isna().any():
                    merged_df[col] = merged_df[col].ffill().bfill()

        # Debug after interpolation
        # print(f"  After interpolation:")
        for col in merged_df.columns:
            if col != 'timestamp':
                missing = merged_df[col].isna().sum()
                print(
                    f"    {col}: {merged_df[col].notna().sum()} non-null values, {missing} missing values")

        # Check final result
        if 'oil' in merged_df.columns and 'natural_gas' in merged_df.columns:
            print(
                f"  Final fuel price dataset: {len(merged_df)} rows with both oil and gas prices")
        elif 'oil' in merged_df.columns:
            print(
                f"  Final fuel price dataset: {len(merged_df)} rows with only oil prices")
        elif 'natural_gas' in merged_df.columns:
            print(
                f"  Final fuel price dataset: {len(merged_df)} rows with only natural gas prices")
        else:
            print("  Warning: Final dataset has no fuel price columns!")

        return merged_df

    def _is_15min_data(self, df):
        """
        Check if the dataframe contains 15-minute interval data.
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            return False

        # Sort by timestamp and get time differences
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()

        if len(time_diffs) == 0:
            return False

        # Get the most common time difference in minutes
        most_common_diff = time_diffs.dt.total_seconds().mode().iloc[0] / 60

        # If the most common difference is around 15 minutes
        return abs(most_common_diff - 15) < 5

    def _resample_to_hourly(self, df, value_columns=None):
        """
        Resample 15-minute data to hourly intervals by averaging.
        """
        if 'timestamp' not in df.columns:
            print("Warning: Cannot resample data without timestamp column")
            return df

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Ensure timestamp is datetime
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

        # If no value columns specified, find numeric columns
        if value_columns is None:
            value_columns = [
                col for col in df_copy.columns
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(df_copy[col])
            ]

        if not value_columns:
            print("Warning: No numeric columns found for resampling")
            return df_copy

        # Set timestamp as index for resampling
        df_copy = df_copy.set_index('timestamp')

        # Keep only the columns we want to resample
        cols_to_keep = value_columns + [
            col for col in df_copy.columns
            if col not in value_columns and col != 'timestamp'
        ]
        df_copy = df_copy[cols_to_keep]

        # Resample to hourly frequency
        resampled = df_copy.resample('h').mean()

        # Reset index to get timestamp back as a column
        resampled = resampled.reset_index()

        return resampled

    def prepare_full_dataset(self, years=None):
        """
        Prepare the full dataset by merging all features from Excel files.

        Args:
            years: List of years to load

        Returns:
            DataFrame with all features ready for modeling
        """
        if years is None:
            years = [2017, 2018, 2019]

        # Load all data components
        # print("Loading electricity price data...")
        prices_df = self.load_electricity_prices(years)

        # print("Loading power system features...")
        power_features = self.load_power_system_features(years)

        # print("Loading import-export data...")
        import_export_df = self.load_import_export(years)

        # print("Loading fuel price data...")
        fuel_prices_df = self.load_fuel_prices(years)

        # Start with prices as the base dataframe
        # print("Merging datasets...")
        full_df = prices_df.copy()

        # Print timestamp ranges for debugging
        # print("\nTimestamp ranges in datasets:")
        self._print_dataset_info(prices_df, "Electricity prices")

        for feature_df in power_features:
            if 'feature_type' in feature_df.columns and len(feature_df) > 0:
                feature_type = feature_df['feature_type'].iloc[0]
                self._print_dataset_info(feature_df, feature_type)

        self._print_dataset_info(import_export_df, "Import/Export")
        self._print_dataset_info(fuel_prices_df, "Fuel prices")

        # Process power system features
        for feature_df in power_features:
            if 'feature_type' in feature_df.columns and len(feature_df) > 0:
                feature_type = feature_df['feature_type'].iloc[0]
                # print(f"  Processing {feature_type} data...")

                if 'feature_type' in feature_df.columns:
                    feature_values = feature_df.drop(columns=['feature_type'])
                else:
                    feature_values = feature_df

                # Merge feature into main dataframe
                if feature_type in feature_values.columns:
                    try:
                        full_df = pd.merge(full_df, feature_values[['timestamp', feature_type]],
                                           on='timestamp', how='left')
                    except Exception as e:
                        print(f"  Error merging {feature_type}: {str(e)}")

        # Add import-export data
        if import_export_df is not None:
            try:
                full_df = pd.merge(full_df, import_export_df,
                                   on='timestamp', how='left')
            except Exception as e:
                print(f"  Error merging import-export data: {str(e)}")

        # Add fuel prices
        if fuel_prices_df is not None:
            try:
                full_df = pd.merge(full_df, fuel_prices_df,
                                   on='timestamp', how='left')
            except Exception as e:
                print(f"  Error merging fuel price data: {str(e)}")

        # Calculate ramps
        print("Calculating ramp features...")
        feature_columns = [
            'load_forecast', 'solar_forecast', 'wind_forecast',
            'total_generation', 'net_import_export'
        ]

        for col in feature_columns:
            if col in full_df.columns:
                full_df[f'{col}_ramp'] = full_df[col].diff()

        # Handle missing values
        initial_rows = len(full_df)
        print("\nMissing values per column:")
        for col in full_df.columns:
            missing = full_df[col].isna().sum()
            percent_missing = (missing / len(full_df)) * \
                100 if len(full_df) > 0 else 0
            print(f"  {col}: {missing} ({percent_missing:.2f}%)")

        # Only require essential columns
        essential_columns = ['timestamp', 'price']
        print(
            f"\nRemoving rows with missing values in essential columns: {essential_columns}")
        full_df = full_df.dropna(subset=essential_columns)

        # Fill missing values in non-essential columns
        numeric_cols = full_df.select_dtypes(include=['number']).columns
        if len(full_df) > 0:
            for col in numeric_cols:
                if col not in essential_columns:
                    missing_before = full_df[col].isna().sum()
                    # Try interpolation first
                    full_df[col] = full_df[col].interpolate(
                        method='linear', limit_direction='both')
                    # If any values still missing, use forward/backward fill
                    if full_df[col].isna().any():
                        full_df[col] = full_df[col].ffill().bfill()
                    missing_after = full_df[col].isna().sum()
                    print(
                        f"  {col}: {missing_before-missing_after} missing values filled")

        final_rows = len(full_df)
        print(
            f"\nRemoved {initial_rows - final_rows} rows with missing values in essential columns")
        print(f"Final dataset shape: {full_df.shape}")

        # Check for net_flow vs net_import_export naming inconsistency
        if 'net_flow' in full_df.columns and 'net_import_export' not in full_df.columns:
            print("Renaming 'net_flow' to 'net_import_export' for consistency")
            full_df = full_df.rename(columns={'net_flow': 'net_import_export'})

        # Final features check
        print("\nFeatures check:")
        expected_features = [
            'timestamp', 'price',
            'load_forecast', 'solar_forecast', 'wind_forecast', 'total_generation',
            'net_import_export', 'oil', 'natural_gas'
        ]

        for feature in expected_features:
            if feature in full_df.columns:
                non_null = full_df[feature].notna().sum()
                percent = (non_null / len(full_df)) * \
                    100 if len(full_df) > 0 else 0
                print(f"  {feature}: {non_null} non-null values ({percent:.2f}%)")
            else:
                print(f"  {feature}: Not found in dataset")

        return full_df

    def _print_dataset_info(self, df, name):
        """Helper to print dataset information"""
        if df is not None and 'timestamp' in df.columns and not df.empty and df['timestamp'].notna().any():
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            print(f"  {name}: {min_date} to {max_date} ({len(df)} rows)")
        else:
            print(f"  {name}: No valid timestamp data")

    def prepare_dataset(self, years=None, use_cached=True, custom_fuel_prices_df=None):
        """
        Elektrizitätspreisvorhersage-Datensatz mit Cache-Funktion und optionalen benutzerdefinierten
        Brennstoffpreisen vorbereiten

        Args:
            years: Liste der Jahre [2017, 2018, 2019]
            use_cached: Wenn True, wird ein vorhandener vorbereiteter Datensatz verwendet
            custom_fuel_prices_df: Optionaler DataFrame mit Brennstoffpreisen

        Returns:
            Aufbereiteter DataFrame
        """
        if years is None:
            years = [2017, 2018, 2019]

        # Cached-Datensatz suchen, wenn gewünscht
        if use_cached:
            cached_path = self._find_latest_prepared_dataset()
            if cached_path:
                print(f"Lade gecachten Datensatz: {cached_path}")
                dataset = pd.read_excel(cached_path)
                dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
                return dataset

        # Standard-Datensatz mit prepare_full_dataset laden
        print(f"Datensatz für Jahre {years} wird erstellt...")
        dataset = self.prepare_full_dataset(years=years)

        # Benutzerdefinierte Brennstoffpreise integrieren falls vorhanden
        if custom_fuel_prices_df is not None:
            print("Benutzerdefinierte Brennstoffpreise werden integriert...")
            dataset = self._integrate_custom_fuel_prices(
                dataset, custom_fuel_prices_df)

        # Datensatz speichern
        self._save_dataset(dataset)

        return dataset

    def _find_latest_prepared_dataset(self):
        """Findet den neuesten vorbereiteten Datensatz im Datenverzeichnis"""
        prepared_files = [file for file in os.listdir(self.data_dir)
                          if file.startswith("prepared_dataset_") and
                          (file.endswith(".csv") or file.endswith(".xlsx"))]

        if prepared_files:
            # Nach Dateiname (enthält Zeitstempel) sortieren
            latest_file = sorted(prepared_files)[-1]
            return os.path.join(self.data_dir, latest_file)
        return None

    def _integrate_custom_fuel_prices(self, dataset, fuel_prices_df):
        """Benutzerdefinierte Brennstoffpreise in den Datensatz integrieren"""
        # Zeitstempel als Datetime-Objekte sicherstellen
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
        fuel_prices_df["timestamp"] = pd.to_datetime(
            fuel_prices_df["timestamp"])

        # Existierende Brennstoffpreisspalten entfernen
        for col in ['oil', 'natural_gas']:
            if col in dataset.columns:
                non_null = dataset[col].notna().sum()
                print(
                    f"Entferne vorhandene '{col}'-Spalte mit {non_null} nicht-leeren Werten")
                dataset = dataset.drop(columns=[col])

        # Nach Zeitstempel zusammenführen
        original_rows = len(dataset)
        dataset = pd.merge(dataset, fuel_prices_df, on="timestamp", how="left")
        print(
            f"Datensatz hat nach Zusammenführung {len(dataset)} Zeilen (vorher {original_rows})")

        return dataset

    def _save_dataset(self, dataset):
        """Speichert den Datensatz mit Zeitstempel"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"prepared_dataset_{timestamp}.xlsx"
        filepath = os.path.join(self.data_dir, filename)

        dataset.to_excel(filepath, index=False)
        print(f"Datensatz gespeichert unter: {filepath}")
        print(
            f"Zeitraum: {dataset['timestamp'].min()} bis {dataset['timestamp'].max()}")
        print(
            f"Enthält {len(dataset)} Datenpunkte mit Features: {dataset.columns.tolist()}")

        return filepath


# Usage example:
# data_loader = ExcelElectricityDataLoader("./data/")
# dataset = data_loader.prepare_full_dataset(years=[2017])
# print(f"Final dataset: {dataset.shape} with columns {dataset.columns.tolist()}")
