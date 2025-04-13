"""
Enhanced timestamp parsing for Excel-based electricity price forecasting.
This module handles various timestamp formats found in electricity market data.
"""

import pandas as pd
import re
from datetime import datetime
import os

def parse_timestamps(df, timestamp_column='timestamp'):
    """
    Enhanced timestamp parser that handles multiple formats.
    
    Args:
        df: DataFrame containing timestamp data
        timestamp_column: Name of the column containing timestamps
        
    Returns:
        DataFrame with properly converted timestamps
    """
    if timestamp_column not in df.columns:
        print(f"Error: DataFrame does not contain column '{timestamp_column}'")
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Get a sample of timestamps for analysis (skipping NaN values)
    valid_samples = df[timestamp_column].dropna().astype(str).tolist()
    sample_values = valid_samples[:5] if valid_samples else []
    
    print(f"Timestamp sample values: {sample_values}")
    
    # Check for empty values
    if not valid_samples:
        print("Warning: No valid timestamp values found to analyze")
        return df
    
    # Determine the format based on sample values
    sample = valid_samples[0]
    
    # Try to identify the format pattern
    if " - " in sample:
        # Format like "01.01.2017 00:00 - 01.01.2017 01:00"
        print("Detected timestamp format with range pattern (e.g., '01.01.2017 00:00 - 01.01.2017 01:00')")
        # Extract just the start timestamp
        df['parsed_timestamp'] = df[timestamp_column].astype(str).apply(extract_start_timestamp)
    elif re.match(r'\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2}', sample):
        # European format (DD.MM.YYYY HH:MM)
        print("Detected European format timestamps (DD.MM.YYYY HH:MM)")
        df['parsed_timestamp'] = df[timestamp_column].astype(str).apply(lambda x: parse_european_format(x))
    elif re.match(r'\d{4}-\d{2}-\d{2}', sample):
        # ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        print("Detected ISO format timestamps (YYYY-MM-DD)")
        df['parsed_timestamp'] = pd.to_datetime(df[timestamp_column], errors='coerce')
    else:
        # Try multiple formats
        print(f"Trying multiple timestamp formats for pattern: '{sample}'")
        df['parsed_timestamp'] = df[timestamp_column].astype(str).apply(try_multiple_formats)
    
    # Check success rate
    success_count = df['parsed_timestamp'].notna().sum()
    total_count = len(df)
    success_rate = success_count / total_count * 100 if total_count > 0 else 0
    
    print(f"Timestamp parsing results: {success_count} valid out of {total_count} ({success_rate:.2f}%)")
    
    if success_count > 0:
        # Sample of successfully parsed values
        valid_samples = df.loc[df['parsed_timestamp'].notna(), 'parsed_timestamp'].head(3).tolist()
        print(f"Sample parsed values: {valid_samples}")
        
        # Replace the original timestamp column
        df[timestamp_column] = df['parsed_timestamp']
    else:
        print("Warning: Failed to parse any timestamps. Check your timestamp format.")
    
    # Drop the temporary column
    df = df.drop(columns=['parsed_timestamp'])
    
    return df

def extract_start_timestamp(timestamp_str):
    """Extract the start timestamp from a range like "01.01.2017 00:00 - 01.01.2017 01:00"."""
    try:
        if not isinstance(timestamp_str, str) or " - " not in timestamp_str:
            return None
        
        # Extract the part before the dash
        start_part = timestamp_str.split(" - ")[0].strip()
        
        # Try to parse it as a datetime
        return parse_european_format(start_part)
    except:
        return None

def parse_european_format(timestamp_str):
    """Parse European format timestamp (DD.MM.YYYY HH:MM)."""
    try:
        if not isinstance(timestamp_str, str):
            return None
        
        # Match the pattern DD.MM.YYYY HH:MM
        match = re.match(r'(\d{2})\.(\d{2})\.(\d{4})\s(\d{2}):(\d{2})', timestamp_str)
        if match:
            day, month, year, hour, minute = match.groups()
            return pd.Timestamp(f"{year}-{month}-{day} {hour}:{minute}:00")
        return None
    except:
        return None

def try_multiple_formats(timestamp_str):
    """Try multiple timestamp formats."""
    if not isinstance(timestamp_str, str):
        return None
    
    formats = [
        '%d.%m.%Y %H:%M',     # 01.01.2017 00:00
        '%Y-%m-%d %H:%M:%S',  # 2017-01-01 00:00:00
        '%Y-%m-%d',           # 2017-01-01
        '%d.%m.%Y',           # 01.01.2017
        '%m/%d/%Y %H:%M',     # 01/01/2017 00:00
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(timestamp_str, format=fmt)
        except:
            continue
    
    # If all specific formats fail, try the flexible parser as a last resort
    try:
        return pd.to_datetime(timestamp_str, errors='coerce')
    except:
        return None

def fix_excel_timestamps(file_path, output_path=None):
    """
    Fix timestamps in an Excel file and save the result.
    
    Args:
        file_path: Path to the Excel file
        output_path: Path to save the fixed file (if None, overwrites the original)
    
    Returns:
        Boolean indicating success
    """
    try:
        print(f"Processing file: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Check if there's a timestamp column
        if 'timestamp' not in df.columns:
            print(f"Error: File {file_path} doesn't have a 'timestamp' column")
            return False
        
        # Parse timestamps
        df_fixed = parse_timestamps(df)
        
        # Determine output path
        if output_path is None:
            output_path = file_path
        
        # Save the fixed file
        df_fixed.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Saved fixed file to: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def fix_all_excel_files(directory):
    """
    Fix timestamps in all Excel files in a directory.
    
    Args:
        directory: Directory containing Excel files
    
    Returns:
        Number of successfully processed files
    """
    # Find all Excel files
    excel_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.xlsx') and os.path.isfile(os.path.join(directory, f))
    ]
    
    if not excel_files:
        print(f"No Excel files found in {directory}")
        return 0
    
    print(f"Found {len(excel_files)} Excel files to process")
    
    # Process each file
    success_count = 0
    for file_path in excel_files:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        if fix_excel_timestamps(file_path):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count} out of {len(excel_files)} files")
    
    return success_count

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix timestamps in Excel files for electricity price forecasting")
    parser.add_argument("directory", help="Directory containing Excel files")
    
    args = parser.parse_args()
    
    fix_all_excel_files(args.directory)