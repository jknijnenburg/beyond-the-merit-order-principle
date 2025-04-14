"""
Specialized script to directly load the oil and gas data from Excel files.

Run this script first to create cleaned fuel price data files that can be used 
by the main model.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Set your data directory
DATA_DIR = "./data/xlsx/"
PLOT_DIR = "./plots/"

def load_oil_prices():
    """Load oil prices directly from Excel file, with explicit column handling"""
    file_path = os.path.join(DATA_DIR, "oil_price_2017.xlsx")
    
    if not os.path.exists(file_path):
        print(f"Oil price file not found: {file_path}")
        return None
    
    print(f"Loading oil price file: {file_path}")
    
    try:
        # First try reading with regular pandas
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Print raw column names for debugging
        print(f"Raw column names: {df.columns.tolist()}")
        
        # If column names look weird, try reading with header=None and assign columns manually
        if len(df.columns) <= 2 or any(len(str(col)) > 30 for col in df.columns):
            print("Column names look problematic, trying with explicit header=None")
            df = pd.read_excel(file_path, header=None, engine='openpyxl')
            
            # Based on your excerpt, the data starts right away, and columns are:
            # timestamp, Erster, Hoch, Tief, Schlusskurs, Stuecke
            if len(df.columns) >= 5:  # Make sure we have enough columns
                df.columns = ['timestamp', 'Erster', 'Hoch', 'Tief', 'Schlusskurs', 'Stuecke']
                print("Assigned column names manually")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Find the price column (should be Schlusskurs)
        price_col = 'Schlusskurs' if 'Schlusskurs' in df.columns else df.columns[4]
        
        # Convert price to numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Create clean data with just timestamp and price columns
        clean_df = df[['timestamp', price_col]].copy()
        clean_df = clean_df.rename(columns={price_col: 'oil'})
        
        # Print debug info
        print(f"Oil price data loaded: {len(clean_df)} rows")
        print(f"Data range: {clean_df['timestamp'].min()} to {clean_df['timestamp'].max()}")
        print(f"Sample data (first 3 rows):")
        print(clean_df.head(3))
        
        # Plot the data to verify
        plt.figure(figsize=(10, 5))
        plt.plot(clean_df['timestamp'], clean_df['oil'])
        plt.title('Oil Prices 2017')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'oil_prices_plot.png'))
        print(f"Created oil price plot at {os.path.join(PLOT_DIR, 'oil_prices_plot.png')}")
        
        # Save clean data
        clean_file = os.path.join(DATA_DIR, "oil_price_clean.xlsx")
        clean_df.to_excel(clean_file, index=False)
        print(f"Saved clean oil price data to {clean_file}")
        
        return clean_df
    
    except Exception as e:
        print(f"Error loading oil price data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_gas_prices():
    """Load natural gas prices directly from Excel file, with explicit column handling"""
    file_path = os.path.join(DATA_DIR, "natural_gas_price_2017.xlsx")
    
    if not os.path.exists(file_path):
        print(f"Natural gas price file not found: {file_path}")
        return None
    
    print(f"Loading natural gas price file: {file_path}")
    
    try:
        # First try reading with regular pandas
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Print raw column names for debugging
        print(f"Raw column names: {df.columns.tolist()}")
        
        # If column names look weird, try reading with header=None and assign columns manually
        if len(df.columns) <= 2 or any(len(str(col)) > 30 for col in df.columns):
            print("Column names look problematic, trying with explicit header=None")
            df = pd.read_excel(file_path, header=None, engine='openpyxl')
            
            # Based on your excerpt, the data starts right away, and columns are:
            # timestamp, Erster, Hoch, Tief, Schlusskurs, Stuecke
            if len(df.columns) >= 5:  # Make sure we have enough columns
                df.columns = ['timestamp', 'Erster', 'Hoch', 'Tief', 'Schlusskurs', 'Stuecke']
                print("Assigned column names manually")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Find the price column (should be Schlusskurs)
        price_col = 'Schlusskurs' if 'Schlusskurs' in df.columns else df.columns[4]
        
        # Convert price to numeric
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Create clean data with just timestamp and price columns
        clean_df = df[['timestamp', price_col]].copy()
        clean_df = clean_df.rename(columns={price_col: 'natural_gas'})
        
        # Print debug info
        print(f"Natural gas price data loaded: {len(clean_df)} rows")
        print(f"Data range: {clean_df['timestamp'].min()} to {clean_df['timestamp'].max()}")
        print(f"Sample data (first 3 rows):")
        print(clean_df.head(3))
        
        # Plot the data to verify
        plt.figure(figsize=(10, 5))
        plt.plot(clean_df['timestamp'], clean_df['natural_gas'])
        plt.title('Natural Gas Prices 2017')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'gas_prices_plot.png'))
        print(f"Created gas price plot at {os.path.join(PLOT_DIR, 'gas_prices_plot.png')}")
        
        # Save clean data
        clean_file = os.path.join(DATA_DIR, "natural_gas_price_clean.xlsx")
        clean_df.to_excel(clean_file, index=False)
        print(f"Saved clean natural gas price data to {clean_file}")
        
        return clean_df
    
    except Exception as e:
        print(f"Error loading natural gas price data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_hourly_fuel_prices(oil_df, gas_df, year=2017):
    """
    Create hourly fuel prices by interpolating daily data
    """
    if oil_df is None and gas_df is None:
        print("No fuel price data available")
        return None
    
    # Start with either oil or gas data, or merge them if both available
    if oil_df is not None and gas_df is not None:
        print("Merging oil and gas price data")
        fuel_df = pd.merge(oil_df, gas_df, on='timestamp', how='outer')
    elif oil_df is not None:
        print("Using only oil price data")
        fuel_df = oil_df.copy()
    elif gas_df is not None:
        print("Using only gas price data")
        fuel_df = gas_df.copy()
    
    # Create a continuous hourly date range for the full year
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31 23:00:00")
    
    print(f"Creating hourly timestamp grid from {start_date} to {end_date}")
    hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
    hourly_df = pd.DataFrame({'timestamp': hourly_range})
    
    # Merge with the price data
    merged_df = pd.merge(hourly_df, fuel_df, on='timestamp', how='left')
    
    # Print info about missing values before interpolation
    for col in merged_df.columns:
        if col != 'timestamp':
            missing = merged_df[col].isna().sum()
            pct_missing = (missing / len(merged_df)) * 100
            print(f"Column {col}: {missing} missing values ({pct_missing:.2f}%)")
    
    # Interpolate missing values
    for col in merged_df.columns:
        if col != 'timestamp':
            merged_df[col] = merged_df[col].interpolate(method='linear', limit_direction='both')
            # If any values still missing (at the edges), use forward/backward fill
            if merged_df[col].isna().any():
                merged_df[col] = merged_df[col].ffill().bfill()
    
    # Print info about missing values after interpolation
    for col in merged_df.columns:
        if col != 'timestamp':
            missing = merged_df[col].isna().sum()
            if missing > 0:
                print(f"Warning: Column {col} still has {missing} missing values after interpolation")
    
    # Save the hourly fuel prices
    hourly_file = os.path.join(DATA_DIR, "hourly_fuel_prices.xlsx")
    merged_df.to_excel(hourly_file, index=False)
    print(f"Saved hourly fuel prices to {hourly_file}")
    
    return merged_df

def process_fuel_prices(years=[2017, 2018, 2019]):
    """Brennstoffpreise verarbeiten und stündliche interpolierte Daten erstellen"""

    print("=== BRENNSTOFFPREISDATEN VERARBEITEN ===\n")

    # Rohdaten laden
    oil_df = load_oil_prices()
    gas_df = load_gas_prices()

    # Stündliche Brennstoffpreise erstellen - ANGEPASST für alle Jahre
    all_data = []
    for year in years:
        print(f"\nVerarbeite Brennstoffpreise für das Jahr {year}...")
        year_df = create_hourly_fuel_prices(oil_df, gas_df, year=year)
        if year_df is not None:
            all_data.append(year_df)
    
    if all_data:
        # Alle Jahre zusammenfügen
        hourly_df = pd.concat(all_data, ignore_index=True)
        print("\nBrennstoffpreisdaten erfolgreich verarbeitet!")
        print(f"Stündlicher Brennstoffpreisdatensatz mit {len(hourly_df)} Zeilen erstellt")
        
        # Daten für zukünftige Verwendung speichern
        fuel_prices_path = os.path.join(DATA_DIR, "processed_fuel_prices.xlsx")
        hourly_df.to_excel(fuel_prices_path, index=False)
        print(f"Verarbeitete Brennstoffpreise gespeichert unter {fuel_prices_path}")
        
        return hourly_df
    else:
        print("\nFehler beim Erstellen der stündlichen Brennstoffpreise")
        return None