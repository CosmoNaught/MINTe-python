#!/usr/bin/env python
"""
Utility script to convert R data files (RDS) to Python-compatible formats.

This script requires rpy2 to be installed:
    pip install rpy2

Usage:
    python convert_rds.py input.rds output.csv
    python convert_rds.py input.rds output.pkl
"""

import argparse
import sys
from pathlib import Path


def convert_rds_to_python(input_path: str, output_path: str) -> None:
    """
    Convert an RDS file to CSV or pickle format.
    
    Parameters
    ----------
    input_path : str
        Path to the input RDS file.
    output_path : str
        Path to the output file (.csv or .pkl/.pickle).
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ImportError:
        print("Error: rpy2 is required to convert RDS files.")
        print("Install it with: pip install rpy2")
        print("\nAlternatively, use R to convert the files:")
        print('  library(readr)')
        print(f'  data <- readRDS("{input_path}")')
        print(f'  write_csv(data, "{output_path}")')
        sys.exit(1)

    # Activate pandas conversion
    pandas2ri.activate()
    
    # Load the RDS file
    base = importr('base')
    readRDS = robjects.r['readRDS']
    
    print(f"Loading: {input_path}")
    r_data = readRDS(input_path)
    
    # Convert to pandas
    try:
        df = pandas2ri.rpy2py(r_data)
    except Exception as e:
        print(f"Error converting to pandas: {e}")
        print("The RDS file may contain a non-dataframe object.")
        sys.exit(1)
    
    # Save based on extension
    output_path = Path(output_path)
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
        print(f"Saved as CSV: {output_path}")
    elif output_path.suffix in ('.pkl', '.pickle'):
        df.to_pickle(output_path)
        print(f"Saved as pickle: {output_path}")
    else:
        print(f"Unknown output format: {output_path.suffix}")
        print("Supported formats: .csv, .pkl, .pickle")
        sys.exit(1)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert R RDS files to Python-compatible formats"
    )
    parser.add_argument("input", help="Input RDS file path")
    parser.add_argument("output", help="Output file path (.csv or .pkl)")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    convert_rds_to_python(args.input, args.output)


if __name__ == "__main__":
    main()
