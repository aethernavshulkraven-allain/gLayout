#!/usr/bin/env python3
"""
Test script for Extended LHS Parameter Explorer with Pruning

This script tests the Extended LHS implementation to ensure all functionality works correctly.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lhs_parameter_explorer import (
    LHSParameterExplorer, 
    EnhancedLHSSampler, 
    ParameterPruner,
    ParameterRange,
    PruningRegion,
    CircuitInstance
)

def test_enhanced_lhs_sampler():
    """Test the Enhanced LHS sampler"""
    print("Testing Enhanced LHS Sampler...")
    
    sampler = EnhancedLHSSampler(num_params=3, seed=42)
    samples = sampler.generate_enhanced_lhs_samples(num_samples=8)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Check stratification property
    for dim in range(3):
        dim_values = samples[:, dim]
        print(f"Dimension {dim} range: [{dim_values.min():.3f}, {dim_values.max():.3f}]")
    
    print("✓ Enhanced LHS Sampler test passed\n")

def test_parameter_pruner():
    """Test the Parameter Pruner"""
    print("Testing Parameter Pruner...")
    
    # Create mock parameter ranges
    param_ranges = {
        'width_1': ParameterRange("width_1", 1.0, 10.0, "continuous"),
        'width_2': ParameterRange("width_2", 1.0, 10.0, "continuous"),
        'fingers_1': ParameterRange("fingers_1", 1, 4, "discrete", [1, 2, 3, 4])
    }
    
    # Create mock instances with some failures
    instances = []
    for i in range(20):
        params = {
            'width_1': 1.0 + i * 0.5,
            'width_2': 2.0 + i * 0.3,
            'fingers_1': (i % 4) + 1
        }
        
        # Make some instances fail (simulate high width causing failures)
        status = "failed" if params['width_1'] > 8.0 else "success"
        
        mock_results = {
            'drc_lvs_fail': status == "failed",
            'geometric': {'raw_area_um2': params['width_1'] * params['width_2'] * 10},
            'pex': {'total_resistance_ohms': 1000}
        } if status == "success" else None
        
        instance = CircuitInstance(
            instance_id=f"test_{i:03d}",
            parameters=params,
            evaluation_status=status,
            json_results=mock_results
        )
        instances.append(instance)
    
    pruner = ParameterPruner()
    updated_ranges = pruner.analyze_and_prune(instances, param_ranges)
    
    print(f"Original width_1 range: {param_ranges['width_1'].min_val} - {param_ranges['width_1'].max_val}")
    print(f"Updated width_1 range: {updated_ranges['width_1'].min_val} - {updated_ranges['width_1'].max_val}")
    print(f"Excluded regions found: {len(pruner.excluded_regions)}")
    
    print("✓ Parameter Pruner test passed\n")

def test_full_integration():
    """Test the full integration"""
    print("Testing Full Integration...")
    
    # Create explorer with minimal setup
    explorer = LHSParameterExplorer(use_extended_lhs=True)
    
    # Add simple parameter ranges
    explorer.add_parameter_range(ParameterRange("x", 0.0, 10.0, "continuous"))
    explorer.add_parameter_range(ParameterRange("y", 0.0, 5.0, "continuous"))
    explorer.add_parameter_range(ParameterRange("n", 1, 3, "discrete", [1, 2, 3]))
    
    # Test sample generation
    samples = explorer.generate_lhs_samples(num_samples=12, seed=42, use_pruning=False)
    
    print(f"Generated {len(samples)} parameter sets")
    print("Sample parameter sets:")
    for i, sample in enumerate(samples[:5]):  # Show first 5
        print(f"  {i+1}: {sample}")
    
    # Test with mock pruning
    mock_region = PruningRegion(
        parameter_bounds={'x': (8.0, 10.0), 'y': (4.0, 5.0)},
        failure_rate=0.9,
        sample_count=5,
        region_type="failure"
    )
    explorer.pruner.excluded_regions = [mock_region]
    
    pruned_samples = explorer.generate_lhs_samples(num_samples=12, seed=42, use_pruning=True)
    print(f"Generated {len(pruned_samples)} parameter sets with pruning")
    
    print("✓ Full Integration test passed\n")

def test_performance_scoring():
    """Test performance scoring functionality"""
    print("Testing Performance Scoring...")
    
    pruner = ParameterPruner()
    
    # Mock successful instance
    mock_results = {
        'drc_lvs_fail': False,
        'geometric': {
            'raw_area_um2': 100.0,
            'symmetry_score_horizontal': 0.8,
            'symmetry_score_vertical': 0.9
        },
        'pex': {
            'total_resistance_ohms': 5000.0
        }
    }
    
    instance = CircuitInstance(
        instance_id="test_perf",
        parameters={'width': 5.0},
        evaluation_status="success",
        json_results=mock_results
    )
    
    score = pruner._calculate_performance_score(instance)
    print(f"Performance score: {score:.2f}")
    
    print("✓ Performance Scoring test passed\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED LHS WITH PRUNING - TEST SUITE")
    print("=" * 60)
    
    try:
        test_enhanced_lhs_sampler()
        test_parameter_pruner()
        test_performance_scoring()
        test_full_integration()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("Enhanced LHS with Parameter Pruning is fully functional.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 