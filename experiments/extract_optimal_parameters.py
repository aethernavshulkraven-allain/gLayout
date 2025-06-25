#!/usr/bin/env python3
"""
Extract Optimal Parameters from FVF Parameter Sweep Results

This script analyzes the parameter exploration results and extracts the most optimal
parameter sets based on different optimization criteria.
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lhs_parameter_explorer import ParameterDatabase, CircuitInstance

class OptimalParameterExtractor:
    """Extract optimal parameters from exploration results"""
    
    def __init__(self, database_path: str = None, csv_path: str = None):
        self.database_path = database_path
        self.csv_path = csv_path
        self.df = None
        self.optimal_results = {}
        self.load_data()
    
    def load_data(self):
        """Load data from database or CSV"""
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"📂 Loading data from CSV: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
        elif self.database_path and os.path.exists(self.database_path):
            print(f"📂 Loading data from database: {self.database_path}")
            db = ParameterDatabase(self.database_path)
            # Export to temporary CSV for analysis
            temp_csv = "temp_optimal_analysis.csv"
            db.export_to_csv(temp_csv)
            if os.path.exists(temp_csv):
                self.df = pd.read_csv(temp_csv)
                os.remove(temp_csv)  # Clean up
            else:
                raise ValueError("No successful instances found in database")
        else:
            raise ValueError("Neither CSV nor database file found")
        
        # Filter to only successful evaluations (DRC/LVS pass)
        if 'drc_lvs_fail' in self.df.columns:
            successful_df = self.df[~self.df['drc_lvs_fail']].copy()
            failed_count = len(self.df) - len(successful_df)
            print(f"✅ Loaded {len(successful_df)} successful instances ({failed_count} failed)")
            self.df = successful_df
        else:
            print(f"✅ Loaded {len(self.df)} instances")
        
        if len(self.df) == 0:
            raise ValueError("No successful instances found for optimization")
    
    def get_parameter_columns(self) -> List[str]:
        """Get list of parameter columns (exclude metadata and outputs)"""
        exclude_prefixes = ['instance_id', 'timestamp', 'drc_', 'lvs_', 'area_', 'symmetry_', 'total_', 'evaluation_status']
        param_cols = []
        for col in self.df.columns:
            if not any(col.startswith(prefix) for prefix in exclude_prefixes):
                param_cols.append(col)
        return param_cols
    
    def extract_optimal_single_objective(self, objective: str, maximize: bool = True) -> Dict:
        """Extract optimal parameters for single objective optimization"""
        if objective not in self.df.columns:
            available_objectives = [col for col in self.df.columns if col in ['area_um2', 'symmetry_horizontal', 'symmetry_vertical']]
            raise ValueError(f"Objective '{objective}' not found. Available: {available_objectives}")
        
        # Find optimal row
        if maximize:
            optimal_idx = self.df[objective].idxmax()
            direction = "maximum"
        else:
            optimal_idx = self.df[objective].idxmin()
            direction = "minimum"
        
        optimal_row = self.df.loc[optimal_idx]
        optimal_value = optimal_row[objective]
        
        # Extract parameters
        param_cols = self.get_parameter_columns()
        optimal_params = {}
        for param in param_cols:
            optimal_params[param] = optimal_row[param]
        
        result = {
            'optimization_type': 'single_objective',
            'objective': objective,
            'direction': direction,
            'optimal_value': optimal_value,
            'parameters': optimal_params,
            'instance_id': optimal_row['instance_id'],
            'full_results': {
                'area_um2': optimal_row.get('area_um2', 'N/A'),
                'symmetry_horizontal': optimal_row.get('symmetry_horizontal', 'N/A'),
                'symmetry_vertical': optimal_row.get('symmetry_vertical', 'N/A'),
                'drc_pass': optimal_row.get('drc_pass', 'N/A'),
                'lvs_pass': optimal_row.get('lvs_pass', 'N/A')
            }
        }
        
        print(f"\n🎯 SINGLE OBJECTIVE OPTIMIZATION: {objective} ({direction})")
        print(f"   Optimal value: {optimal_value:.4f}")
        print(f"   Instance ID: {optimal_row['instance_id']}")
        
        return result
    
    def extract_optimal_composite_score(self, weights: Dict[str, float] = None) -> Dict:
        """Extract optimal parameters using composite scoring"""
        if weights is None:
            # Default weights: minimize area, maximize symmetries
            weights = {
                'area_um2': -1.0,           # Minimize area (negative weight)
                'symmetry_horizontal': 1.0,  # Maximize horizontal symmetry
                'symmetry_vertical': 1.0     # Maximize vertical symmetry
            }
        
        # Normalize objectives to [0, 1] scale
        df_norm = self.df.copy()
        score_components = []
        
        for objective, weight in weights.items():
            if objective not in self.df.columns:
                print(f"⚠️ Warning: {objective} not found in data, skipping")
                continue
            
            obj_values = self.df[objective]
            
            # Normalize to [0, 1]
            if weight > 0:  # Maximize
                normalized = (obj_values - obj_values.min()) / (obj_values.max() - obj_values.min())
            else:  # Minimize (negative weight)
                normalized = (obj_values.max() - obj_values) / (obj_values.max() - obj_values.min())
            
            weighted_score = abs(weight) * normalized
            score_components.append(weighted_score)
            df_norm[f'{objective}_normalized'] = normalized
            df_norm[f'{objective}_weighted'] = weighted_score
        
        if not score_components:
            raise ValueError("No valid objectives found for composite scoring")
        
        # Calculate composite score
        composite_score = sum(score_components)
        df_norm['composite_score'] = composite_score
        
        # Find optimal
        optimal_idx = composite_score.idxmax()
        optimal_row = df_norm.loc[optimal_idx]
        optimal_score = optimal_row['composite_score']
        
        # Extract parameters
        param_cols = self.get_parameter_columns()
        optimal_params = {}
        for param in param_cols:
            optimal_params[param] = optimal_row[param]
        
        result = {
            'optimization_type': 'composite_score',
            'weights': weights,
            'optimal_score': optimal_score,
            'parameters': optimal_params,
            'instance_id': optimal_row['instance_id'],
            'full_results': {
                'area_um2': optimal_row.get('area_um2', 'N/A'),
                'symmetry_horizontal': optimal_row.get('symmetry_horizontal', 'N/A'),
                'symmetry_vertical': optimal_row.get('symmetry_vertical', 'N/A'),
                'drc_pass': optimal_row.get('drc_pass', 'N/A'),
                'lvs_pass': optimal_row.get('lvs_pass', 'N/A')
            },
            'score_breakdown': {}
        }
        
        # Add score breakdown
        for objective, weight in weights.items():
            if objective in self.df.columns:
                result['score_breakdown'][objective] = {
                    'value': optimal_row[objective],
                    'normalized': optimal_row.get(f'{objective}_normalized', 'N/A'),
                    'weighted': optimal_row.get(f'{objective}_weighted', 'N/A'),
                    'weight': weight
                }
        
        print(f"\n🎯 COMPOSITE SCORE OPTIMIZATION")
        print(f"   Optimal score: {optimal_score:.4f}")
        print(f"   Instance ID: {optimal_row['instance_id']}")
        print(f"   Weights: {weights}")
        
        return result
    
    def extract_pareto_optimal(self, objectives: List[str] = None) -> List[Dict]:
        """Extract Pareto optimal parameter sets"""
        if objectives is None:
            objectives = ['area_um2', 'symmetry_horizontal']
        
        # Check available objectives
        available_objectives = [obj for obj in objectives if obj in self.df.columns]
        if len(available_objectives) < 2:
            print(f"⚠️ Warning: Only {len(available_objectives)} objectives available for Pareto analysis")
            available_objectives = [col for col in self.df.columns if col in ['area_um2', 'symmetry_horizontal', 'symmetry_vertical']][:2]
        
        objectives = available_objectives
        print(f"\n🎯 PARETO OPTIMAL EXTRACTION")
        print(f"   Objectives: {objectives}")
        
        # Find Pareto front
        pareto_indices = []
        df_work = self.df.copy()
        
        for i, row in df_work.iterrows():
            dominated = False
            for j, other_row in df_work.iterrows():
                if i != j:
                    # Check if current solution is dominated
                    # Assume: minimize area_um2, maximize symmetry metrics
                    better_or_equal = True
                    strictly_better = False
                    
                    for obj in objectives:
                        if obj == 'area_um2':
                            # Minimize area
                            if row[obj] > other_row[obj]:
                                better_or_equal = False
                                break
                            elif row[obj] < other_row[obj]:
                                strictly_better = True
                        else:
                            # Maximize symmetry
                            if row[obj] < other_row[obj]:
                                better_or_equal = False
                                break
                            elif row[obj] > other_row[obj]:
                                strictly_better = True
                    
                    if better_or_equal and strictly_better:
                        dominated = True
                        break
            
            if not dominated:
                pareto_indices.append(i)
        
        print(f"   Found {len(pareto_indices)} Pareto optimal solutions")
        
        # Extract Pareto optimal parameter sets
        pareto_results = []
        param_cols = self.get_parameter_columns()
        
        for idx in pareto_indices:
            row = df_work.loc[idx]
            
            optimal_params = {}
            for param in param_cols:
                optimal_params[param] = row[param]
            
            result = {
                'optimization_type': 'pareto_optimal',
                'objectives': objectives,
                'parameters': optimal_params,
                'instance_id': row['instance_id'],
                'objective_values': {obj: row[obj] for obj in objectives},
                'full_results': {
                    'area_um2': row.get('area_um2', 'N/A'),
                    'symmetry_horizontal': row.get('symmetry_horizontal', 'N/A'),
                    'symmetry_vertical': row.get('symmetry_vertical', 'N/A'),
                    'drc_pass': row.get('drc_pass', 'N/A'),
                    'lvs_pass': row.get('lvs_pass', 'N/A')
                }
            }
            pareto_results.append(result)
        
        return pareto_results
    
    def extract_all_optimal_sets(self) -> Dict:
        """Extract all types of optimal parameter sets"""
        results = {}
        
        # 1. Single objective optimizations
        single_objectives = [
            ('area_um2', False),  # Minimize area
            ('symmetry_horizontal', True),  # Maximize horizontal symmetry
            ('symmetry_vertical', True)     # Maximize vertical symmetry
        ]
        
        results['single_objective'] = {}
        for obj, maximize in single_objectives:
            if obj in self.df.columns:
                try:
                    results['single_objective'][obj] = self.extract_optimal_single_objective(obj, maximize)
                except Exception as e:
                    print(f"⚠️ Warning: Could not optimize {obj}: {e}")
        
        # 2. Composite score optimization
        try:
            results['composite'] = self.extract_optimal_composite_score()
        except Exception as e:
            print(f"⚠️ Warning: Could not perform composite optimization: {e}")
        
        # 3. Pareto optimal sets
        try:
            results['pareto'] = self.extract_pareto_optimal()
        except Exception as e:
            print(f"⚠️ Warning: Could not extract Pareto optimal sets: {e}")
        
        return results
    
    def save_optimal_parameters(self, results: Dict, output_file: str = "optimal_parameters.json"):
        """Save optimal parameters to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"\n💾 Optimal parameters saved to: {output_file}")
    
    def print_summary(self, results: Dict):
        """Print summary of optimal parameter sets"""
        print("\n" + "="*80)
        print("🏆 OPTIMAL PARAMETER EXTRACTION SUMMARY")
        print("="*80)
        
        # Single objective results
        if 'single_objective' in results:
            print("\n📊 SINGLE OBJECTIVE OPTIMIZATION:")
            for obj, result in results['single_objective'].items():
                print(f"   {obj}: {result['optimal_value']:.4f} (Instance: {result['instance_id']})")
        
        # Composite score result
        if 'composite' in results:
            comp = results['composite']
            print(f"\n🎯 COMPOSITE SCORE OPTIMIZATION:")
            print(f"   Score: {comp['optimal_score']:.4f} (Instance: {comp['instance_id']})")
            print(f"   Area: {comp['full_results']['area_um2']:.2f} um²")
            print(f"   Symmetry H: {comp['full_results']['symmetry_horizontal']:.3f}")
            print(f"   Symmetry V: {comp['full_results']['symmetry_vertical']:.3f}")
        
        # Pareto results
        if 'pareto' in results and results['pareto']:
            print(f"\n⚡ PARETO OPTIMAL SOLUTIONS: {len(results['pareto'])} found")
            print("   Top 3 by area efficiency:")
            pareto_sorted = sorted(results['pareto'], key=lambda x: x['objective_values'].get('area_um2', float('inf')))
            for i, result in enumerate(pareto_sorted[:3]):
                obj_vals = result['objective_values']
                print(f"   #{i+1}: Area={obj_vals.get('area_um2', 'N/A'):.2f}, Sym_H={obj_vals.get('symmetry_horizontal', 'N/A'):.3f} (ID: {result['instance_id']})")

def main():
    """Main function for optimal parameter extraction"""
    parser = argparse.ArgumentParser(description='Extract optimal FVF parameters from exploration results')
    parser.add_argument('--database', type=str, help='Path to parameter database (.pkl file)')
    parser.add_argument('--csv', type=str, help='Path to exploration results CSV file')
    parser.add_argument('--output', type=str, default='optimal_parameters.json', 
                       help='Output file for optimal parameters (JSON)')
    parser.add_argument('--objective', type=str, choices=['area', 'symmetry_h', 'symmetry_v', 'composite'], 
                       default='composite', help='Optimization objective')
    
    args = parser.parse_args()
    
    # Auto-detect files if not specified
    if not args.database and not args.csv:
        # Look for recent database files
        db_files = list(Path('.').glob('parameter_database_fvf_*.pkl'))
        csv_files = list(Path('.').glob('lhs_outputs*/exploration_results.csv'))
        
        if db_files:
            args.database = str(max(db_files, key=os.path.getctime))
            print(f"🔍 Auto-detected database: {args.database}")
        elif csv_files:
            args.csv = str(max(csv_files, key=os.path.getctime))
            print(f"🔍 Auto-detected CSV: {args.csv}")
        else:
            print("❌ No database or CSV files found. Please run parameter exploration first.")
            return
    
    try:
        # Create extractor
        extractor = OptimalParameterExtractor(args.database, args.csv)
        
        # Extract optimal parameters
        if args.objective == 'composite':
            results = extractor.extract_all_optimal_sets()
        else:
            # Single objective optimization
            obj_map = {
                'area': ('area_um2', False),
                'symmetry_h': ('symmetry_horizontal', True),
                'symmetry_v': ('symmetry_vertical', True)
            }
            obj_name, maximize = obj_map[args.objective]
            results = {'single_objective': {obj_name: extractor.extract_optimal_single_objective(obj_name, maximize)}}
        
        # Print summary
        extractor.print_summary(results)
        
        # Save results
        extractor.save_optimal_parameters(results, args.output)
        
        print(f"\n✅ Optimal parameter extraction complete!")
        print(f"📁 Results saved to: {args.output}")
        
        # Print usage instructions
        if 'composite' in results:
            comp_params = results['composite']['parameters']
            print(f"\n🚀 TO USE THE OPTIMAL PARAMETERS:")
            print(f"   Device Type: {comp_params.get('device_type', 'N/A')}")
            print(f"   Placement: {comp_params.get('placement', 'N/A')}")
            print(f"   Widths: [{comp_params.get('width_1', 'N/A'):.3f}, {comp_params.get('width_2', 'N/A'):.3f}]")
            print(f"   Lengths: [{comp_params.get('length_1', 'N/A'):.3f}, {comp_params.get('length_2', 'N/A'):.3f}]")
            print(f"   Fingers: [{comp_params.get('fingers_1', 'N/A')}, {comp_params.get('fingers_2', 'N/A')}]")
            print(f"   Multipliers: [{comp_params.get('multipliers_1', 'N/A')}, {comp_params.get('multipliers_2', 'N/A')}]")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 