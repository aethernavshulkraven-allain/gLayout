#!/usr/bin/env python3
"""
Simple FVF Optimization Runner

Quick demo script to run FVF parameter optimization using Extended LHS sampling.
This finds the best possible FVF parameter combinations through intelligent sweeps.
"""

import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# Fix PDK_ROOT path to point to the correct Sky130 PDK installation
os.environ['PDK_ROOT'] = '/Applications/conda/miniconda3/envs/openfasoc-env/share/pdk'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the glayout src directory to Python path
sys.path.append('/Users/adityakak/glayout_experiments/glayout/src')

from fvf_optimization import FVFOptimizer

def cleanup_files_to_results():
    """Move all generated files to results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to move
    patterns = [
        "fvf_*.drc.rpt",
        "fvf_*.nodes", 
        "fvf_*.res.ext",
        "fvf_*.sim",
        "fvf_*.lvs.rpt",
        "fvf_*.gds",
        "evaluation_*.log"
    ]
    
    moved_count = 0
    for pattern in patterns:
        for file_path in Path(".").glob(pattern):
            try:
                shutil.move(str(file_path), str(results_dir / file_path.name))
                moved_count += 1
            except Exception as e:
                print(f"Warning: Could not move {file_path}: {e}")
    
    if moved_count > 0:
        print(f"Moved {moved_count} files to {results_dir}")
    
    return results_dir

def run_quick_optimization():
    """Run a quick FVF optimization demo"""
    
    print("🔬 FVF PARAMETER OPTIMIZATION DEMO")
    print("Using Extended Latin Hypercube Sampling with Intelligent Pruning")
    print("=" * 70)
    
    # Create optimizer with conservative parameters for demo
    optimizer = FVFOptimizer(exploration_type="conservative")
    
    # Run optimization with a reasonable number of samples
    print("\n🚀 Starting optimization...")
    print("This will systematically explore FVF parameter space to find optimal designs.")
    
    results = optimizer.run_optimization(
        total_samples=30,  # Start with 30 samples for demo
        batch_size=10      # Process in batches of 10
    )
    
    if results.get("status") == "success":
        print("\n✅ OPTIMIZATION COMPLETED!")
        
        # Display key results
        summary = results.get("summary", {})
        best_designs = results.get("best_designs", {})
        
        print(f"\n📈 OPTIMIZATION RESULTS:")
        print(f"   • Total designs evaluated: {summary.get('total_evaluations', 0)}")
        print(f"   • Successful designs: {summary.get('successful_designs', 0)}")
        print(f"   • Success rate: {summary.get('success_rate', 0):.1f}%")
        print(f"   • Optimization time: {summary.get('optimization_duration_seconds', 0):.1f} seconds")
        
        if best_designs.get('best_overall'):
            best = best_designs['best_overall']
            print(f"\n🏆 BEST DESIGN FOUND:")
            print(f"   • Instance ID: {best.get('instance_id', 'N/A')}")
            print(f"   • Composite Score: {best.get('composite_score', 0):.2f}/100")
            print(f"   • Area: {best.get('area_um2', 0):.1f} μm²")
            print(f"   • Symmetry Score: {best.get('symmetry_score', 0):.3f}")
            print(f"   • Area Efficiency: {best.get('area_efficiency', 0):.2f}")
            
            print(f"\n⚙️  OPTIMAL PARAMETERS:")
            params = best.get('parameters', {})
            for param_name, value in params.items():
                if isinstance(value, float):
                    print(f"   • {param_name}: {value:.3f}")
                else:
                    print(f"   • {param_name}: {value}")
        
        # Show top 3 designs
        top_designs = best_designs.get('top_10_overall', [])[:3]
        if len(top_designs) > 1:
            print(f"\n🥉 TOP 3 DESIGNS:")
            for i, design in enumerate(top_designs):
                print(f"   {i+1}. Score: {design.get('composite_score', 0):.1f}, Area: {design.get('area_um2', 0):.1f} μm²")
        
        print(f"\n📁 All results saved in: {optimizer.output_dir}")
        print("   • optimization_results.json - Complete detailed results")
        print("   • optimization_summary.txt - Human-readable summary")
        print("   • top_designs.csv - Best performing designs")
        print("   • gds_files/ - GDS files for all evaluated designs")
        
        # Suggest next steps
        print(f"\n🎯 NEXT STEPS:")
        print("   1. Review the detailed results in the output directory")
        print("   2. Examine the GDS files of top-performing designs")
        print("   3. Run with more samples for comprehensive optimization:")
        print("      python run_fvf_optimization.py --samples 100 --exploration extended")
        
        return True
        
    else:
        print(f"\n❌ OPTIMIZATION FAILED: {results.get('reason', 'Unknown error')}")
        print("Check the evaluation setup and parameter ranges.")
        return False

def run_extended_optimization():
    """Run a more comprehensive optimization"""
    
    print("🔬 EXTENDED FVF PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    optimizer = FVFOptimizer(exploration_type="extended")
    
    results = optimizer.run_optimization(
        total_samples=100,  # More comprehensive sweep
        batch_size=20
    )
    
    return results.get("status") == "success"

def main():
    """Main optimization runner"""
    print("🚀 Starting FVF Circuit Optimization with Extended LHS")
    print("=" * 60)
    
    try:
        # Initialize optimizer
        optimizer = FVFOptimizer(
            output_dir="results/current_run"
        )
        
        # Run optimization
        results = optimizer.run_optimization()
        
        print("\n" + "=" * 60)
        print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        if results.get("status") == "success":
            best_designs = results.get("best_designs", {})
            print(f"Found {len(best_designs.get('top_designs', []))} high-quality designs")
            
            # Print summary of best design
            if best_designs.get("best_overall"):
                best = best_designs["best_overall"]
                print(f"\nBest Overall Design: {best['instance_id']}")
                print(f"  Area: {best['area_um2']:.1f} μm²")
                print(f"  Composite Score: {best['composite_score']:.1f}")
                print(f"  DRC Pass: {best['drc_pass']}")
                print(f"  LVS Pass: {best['lvs_pass']}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup files regardless of success/failure
        print("\n" + "=" * 60)
        print("🧹 CLEANING UP FILES")
        print("=" * 60)
        results_dir = cleanup_files_to_results()
        print(f"All generated files moved to: {results_dir}")

if __name__ == "__main__":
    main() 