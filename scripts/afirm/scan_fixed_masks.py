"""
Scan downloaded AFIRM data and parse out subject IDs and sessions
"""
import glob
import os
import re
import sys
import datetime
import csv
import shutil

SUMMARY = "/home/bbzmsc/code/fsort/scripts/afirm/afirm_subjects_summary_20251021.csv"
FIXED_MASKS = "/spmstore/project/RenalMRI/afirm/DEFINITIVETKVMASKS"
FIXED_MASKS_OUT = "/spmstore/project/RenalMRI/afirm/DEFINITIVETKVMASKS_RENAMED"

# Function to read the CSV file and convert it into a list of dictionaries with column indexes as keys
def read_summary_to_dicts(summary_path):
    data = []
    try:
        with open(summary_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Create a dictionary where keys are column indexes and values are the row values
                data.append({index: value for index, value in enumerate(row)})
    except FileNotFoundError:
        print(f"Error: File not found at {summary_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

# Function to scan the FIXED_MASKS directory for NIfTI files with 'mask' in the filename
def find_mask_files(directory):
    mask_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if "mask" in file.lower() and file.endswith((".nii", ".nii.gz")):
                dir_name = os.path.basename(root)  # Get the name of the directory containing the file
                mask_files[dir_name] = os.path.join(root, file)
    return mask_files

# Example usage
if __name__ == "__main__":
    # Read the summary CSV file
    summary_data = read_summary_to_dicts(SUMMARY)
    print(f"Loaded {len(summary_data)} rows from the summary CSV file.")

    # Find mask files in the FIXED_MASKS directory
    mask_files = find_mask_files(FIXED_MASKS)
    print(f"Found {len(mask_files)} mask files:")

    # Ensure the output directory exists
    os.makedirs(FIXED_MASKS_OUT, exist_ok=True)

    # Track matched rows
    matched_rows = set()

    # Iterate through the mask files and find matches in the CSV data
    for dir_name, mask_file in mask_files.items():
        matches = []
        last_match = None
        for row in summary_data:
            # Check if the directory name matches columns 0 and 1 (case insensitive, without underscores)
            if row[0].lower() + "_" in dir_name.lower() and row[6] == "BASELINE":
                # Remove underscores from row[3] and pad column 4 as a three-digit integer
                padded_col4 = f"{int(row[4]):03}"
                matches.append(f"{row[3]}_{padded_col4}")
                matched_rows.add((row[0], row[1]))  # Track matched rows by columns 0 and 1

        if len(matches) == 1:
            # Copy the mask file to the output directory with the new name
            matched_id = matches[0]
            output_file = os.path.join(FIXED_MASKS_OUT, f"{matched_id}_V1.nii.gz")
            shutil.copy(mask_file, output_file)
            print(f"Copied: {mask_file} -> {output_file}")
        elif len(matches) > 1:
            # Print a warning if multiple matches are found
            print(f"WARNING: {mask_file} -> MULTIPLE MATCHES FOUND: {matches}")
        else:
            # Print the mask file path and "NO MATCH" if no match is found
            print(f"NO MATCH: {mask_file}")

    # Check for unmatched rows in the CSV
    unmatched_rows = [
        row for row in summary_data
        if row[6] == "BASELINE" and (row[0], row[1]) not in matched_rows
    ]

    if unmatched_rows:
        print("\nWARNING: The following rows in the CSV file with 'BASELINE' were not matched:")
        for row in unmatched_rows:
            print(f"Unmatched row: {row}")
    else:
        print("\nAll 'BASELINE' rows in the CSV file were matched.")
