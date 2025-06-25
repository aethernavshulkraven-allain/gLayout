#!/usr/bin/env python3
"""
LHS Parameter Exploration Runner Script

This script provides an easy interface to run parameter exploration 
for circuit layouts using Latin Hypercube Sampling.

Usage:
    python run_lhs_exploration.py --circuit fvf --samples 50 --exploration conservative
"""

import argparse
import sys
import os
import shutil
import glob
from datetime import datetime

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lhs_parameter_explorer import LHSParameterExplorer
from parameter_config import get_parameter_ranges, print_parameter_summary

def get_next_run_number():
    """Find the next available run number in results directory"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    
    existing_runs = []
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and item.startswith("run-"):
            try:
                run_num = int(item.split("-")[1])
                existing_runs.append(run_num)
            except (IndexError, ValueError):
                continue
    
    return max(existing_runs) + 1 if existing_runs else 1

def setup_run_directory(run_number):
    """Create and return the path to the run directory"""
    run_dir = f"results/run-{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def move_files_to_run_directory(run_dir, circuit_name):
    """Move all generated files to the run directory"""
    print(f"\nOrganizing output files into {run_dir}/...")
    
    # Patterns for files to move
    file_patterns = [
        f"{circuit_name}*.gds",
        f"{circuit_name}*.drc.rpt", 
        f"{circuit_name}*.nodes",
        f"{circuit_name}*.res.ext",
        f"{circuit_name}*.sim",
        f"{circuit_name}*.lvs.rpt",
        f"{circuit_name}*.pex.spice",
        "parameter_database_*.pkl",
        "evaluation*.log",
        "lhs_outputs*"
    ]
    
    moved_files = []
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                dest_path = os.path.join(run_dir, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
                moved_files.append(file_path)
                print(f"  Moved: {file_path} -> {dest_path}")
            elif os.path.isdir(file_path):
                dest_path = os.path.join(run_dir, os.path.basename(file_path))
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(file_path, dest_path)
                moved_files.append(file_path)
                print(f"  Moved: {file_path}/ -> {dest_path}/")
    
    print(f"  Total files/directories moved: {len(moved_files)}")
    return moved_files

def main():
    parser = argparse.ArgumentParser(description='Run LHS parameter exploration for circuit layouts')
    
    parser.add_argument('--circuit', type=str, default='fvf', 
                       choices=['fvf', 'opamp'],
                       help='Circuit type to explore (default: fvf)')
    
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of LHS samples to generate (default: 20)')
    
    parser.add_argument('--exploration', type=str, default='conservative',
                       choices=['conservative', 'default', 'extended'],
                       help='Exploration level - affects parameter ranges (default: conservative)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: lhs_outputs_TIMESTAMP)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show parameter ranges without running exploration')
    
    parser.add_argument('--database', type=str, default=None,
                       help='Database file path (default: parameter_database.pkl)')
    
    args = parser.parse_args()
    
    # Get the next run number and setup run directory
    run_number = get_next_run_number()
    run_dir = setup_run_directory(run_number)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"lhs_outputs_{args.circuit}_{args.exploration}_{timestamp}"
    else:
        output_dir = args.output_dir
    
    # Setup database path
    if args.database is None:
        database_path = f"parameter_database_{args.circuit}_{args.exploration}.pkl"
    else:
        database_path = args.database
    
    print("=" * 60)
    print("LHS Parameter Exploration")
    print("=" * 60)
    print(f"Circuit Type: {args.circuit}")
    print(f"Exploration Level: {args.exploration}")
    print(f"Number of Samples: {args.samples}")
    print(f"Random Seed: {args.seed}")
    print(f"Output Directory: {output_dir}")
    print(f"Database: {database_path}")
    print(f"Run Number: {run_number}")
    print(f"Final Results Directory: {run_dir}")
    print("=" * 60)
    
    # Get parameter ranges
    try:
        param_ranges = get_parameter_ranges(args.circuit, args.exploration)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Print parameter summary
    print_parameter_summary(param_ranges)
    
    if args.dry_run:
        print("\nDry run completed. No exploration was performed.")
        return 0
    
    # Confirm before proceeding
    print(f"\nReady to run {args.samples} evaluations.")
    print("This may take a while depending on the number of samples.")
    
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Exploration cancelled.")
        return 0
    
    # Create explorer
    explorer = LHSParameterExplorer(database_path)
    explorer.output_dir = output_dir
    
    # Setup parameter ranges
    for param_range in param_ranges.values():
        explorer.add_parameter_range(param_range)
    
    print(f"\nStarting exploration...")
    
    try:
        # Run exploration
        results = explorer.run_exploration(
            num_samples=args.samples,
            seed=args.seed,
            save_frequency=max(1, args.samples // 10)  # Save every 10% of progress
        )
        
        # Print final results
        successful = len([r for r in results if r.evaluation_status == "success"])
        failed = len(results) - successful
        
        print("\n" + "=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        
        # Move all generated files to the run directory
        moved_files = move_files_to_run_directory(run_dir, args.circuit)
        
        print(f"\nResults organized in: {run_dir}/")
        print(f"- GDS files: {args.circuit}_*.gds")
        print(f"- Database: {database_path}")
        print(f"- CSV export: exploration_results.csv")
        print(f"- Log files: evaluation*.log")
        
        if successful > 0:
            print(f"\nYou can now analyze the results using the CSV file or")
            print(f"load the database programmatically for further analysis.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExploration interrupted by user.")
        print("Partial results have been saved to the database.")
        # Move any generated files to run directory
        move_files_to_run_directory(run_dir, args.circuit)
        print(f"Partial results moved to: {run_dir}")
        return 1
    
    except Exception as e:
        print(f"\nError during exploration: {e}")
        print("Check the error logs and try again.")
        # Move any generated files to run directory for debugging
        move_files_to_run_directory(run_dir, args.circuit)
        print(f"Generated files moved to: {run_dir}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 