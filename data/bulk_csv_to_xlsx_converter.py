import pandas as pd
import argparse
import os
import glob # Used for finding files matching a pattern

def convert_csv_to_xlsx(csv_filepath, xlsx_filepath):
    """
    Reads a comma-separated CSV file, renames the first column to 'timestamp',
    and saves its content to an XLSX file.

    Args:
        csv_filepath (str): The path to the input CSV file.
        xlsx_filepath (str): The path where the output XLSX file should be saved.

    Returns:
        bool: True if conversion (including rename attempt) was successful or handled, False otherwise.
    """
    # Basic check if the source file exists before attempting to read
    if not os.path.isfile(csv_filepath):
        print(f"Skipping: Input file not found or is not a file: '{csv_filepath}'")
        return False # Indicate failure/skip

    try:
        print(f"Reading CSV file: '{csv_filepath}'...")
        # Read the CSV file using pandas, explicitly stating the comma separator
        df = pd.read_csv(csv_filepath, sep=',')

        # --- Modification: Rename the first column ---
        new_column_name = "timestamp"
        if not df.empty and len(df.columns) > 0:
            # Get the current name of the first column
            original_first_column_name = df.columns[0]

            # Check if renaming is actually needed
            if original_first_column_name != new_column_name:
                print(f"Renaming first column from '{original_first_column_name}' to '{new_column_name}'...")
                # Use the rename method. inplace=True modifies the DataFrame directly.
                df.rename(columns={original_first_column_name: new_column_name}, inplace=True)
            else:
                print(f"First column is already named '{new_column_name}'. No rename needed.")

        elif df.empty and len(df.columns) == 0:
            # Handles CSVs that are truly empty (no headers, no data)
             print(f"Warning: CSV file '{csv_filepath}' is completely empty. Cannot rename columns. Creating empty XLSX file.")
        elif df.empty:
             # Handles CSVs that might have headers but no data rows
            print(f"Warning: CSV file '{csv_filepath}' has headers but no data. First column name checked/renamed if headers exist.")
            # If headers exist (len(df.columns) > 0), the rename logic above would have run.
            # If headers don't exist (len(df.columns) == 0), this case is covered above.
            # If headers exist but first is already 'timestamp', it's handled.
        else:
            # Should not typically happen if df is not empty, but included for completeness
             print(f"Warning: DataFrame is not empty but has no columns (unexpected state). Cannot rename.")
        # --- End of Modification ---


        print(f"Writing XLSX file: '{xlsx_filepath}'...")
        # Write the potentially modified DataFrame to an Excel file
        df.to_excel(xlsx_filepath, index=False, engine='openpyxl')

        print(f"Successfully converted '{os.path.basename(csv_filepath)}' to '{os.path.basename(xlsx_filepath)}' (first column: '{new_column_name}')")
        return True # Indicate success

    except pd.errors.EmptyDataError:
        # This specific pandas error might occur if read_csv fails on an empty file
        print(f"Warning: CSV file '{csv_filepath}' triggered EmptyDataError. Creating an empty XLSX file.")
        pd.DataFrame().to_excel(xlsx_filepath, index=False, engine='openpyxl')
        print(f"Successfully created empty XLSX file: '{xlsx_filepath}'")
        return True # Still counts as processed

    except Exception as e:
        # Catch other potential errors during read/rename/write
        print(f"Error converting file '{csv_filepath}': {e}")
        return False # Indicate failure

# The main execution block (__name__ == "__main__") remains unchanged
# as it handles file discovery and passes paths to the modified function above.

if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Convert all comma-separated CSV files (.csv) in a specified folder to Excel files (.xlsx), renaming the first column to 'timestamp'."
    )

    # Required argument: input FOLDER path
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing the input comma-separated CSV files."
    )

    # Optional argument: output FOLDER path
    parser.add_argument(
        "-o", "--output",
        dest="output_folder", # Give the destination variable a clear name
        help="Path to the folder where output XLSX files should be saved (optional). "
             "If not provided, XLSX files will be saved in the same folder as the input CSVs."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    input_dir = args.input_folder
    output_dir = args.output_folder

    # --- Input Folder Validation ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input path '{input_dir}' is not a valid directory.")
        exit(1) # Exit script with an error code

    # --- Output Folder Handling ---
    if output_dir:
        # If an output directory is specified, create it if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents error if dir exists
            print(f"Outputting XLSX files to: '{output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            exit(1)
    else:
        # If no output directory is specified, use the input directory
        output_dir = input_dir
        print(f"Outputting XLSX files to the input directory: '{input_dir}'")

    # --- Find and Process CSV Files ---
    # Use glob to find all files ending with .csv (case-insensitive)
    search_pattern_lower = os.path.join(input_dir, '*.csv')
    search_pattern_upper = os.path.join(input_dir, '*.CSV') # Include uppercase extension
    csv_files = glob.glob(search_pattern_lower) + glob.glob(search_pattern_upper)
    csv_files = list(set(csv_files)) # Remove duplicates if OS is case-insensitive
    csv_files = [f for f in csv_files if os.path.isfile(f)] # Ensure they are files

    if not csv_files:
        print(f"\nNo CSV files found in '{input_dir}'.")
        exit(0)

    print(f"\nFound {len(csv_files)} CSV file(s) to process...")
    converted_count = 0
    failed_count = 0

    for csv_file_path in csv_files:
        print("-" * 30) # Separator for clarity between files
        # Extract the base name (filename without extension)
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        # Construct the corresponding XLSX filename
        xlsx_filename = base_name + ".xlsx"
        # Construct the full path for the output XLSX file
        xlsx_file_path = os.path.join(output_dir, xlsx_filename)

        # Call the conversion function (which now includes the renaming logic)
        if convert_csv_to_xlsx(csv_file_path, xlsx_file_path):
            converted_count += 1
        else:
            failed_count += 1

    # --- Final Summary ---
    print("\n" + "=" * 30)
    print("Bulk Conversion Summary:")
    print(f"  Total CSV files found: {len(csv_files)}")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Failed/Skipped: {failed_count}")
    print("=" * 30)