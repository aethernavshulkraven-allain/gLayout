#!/usr/bin/env python3
"""
Test script for full FVF parameter sweeping functionality.
Verifies that all parameters are properly swept including device_type and placement.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the glayout src directory to Python path
sys.path.append('/Users/adityakak/glayout_experiments/glayout/src')

from parameter_config import get_parameter_ranges, print_parameter_summary
from lhs_parameter_explorer import LHSParameterExplorer, ParameterRange

def test_parameter_configurations():
    """Test that all parameter configurations are properly defined"""
    print("🧪 TESTING PARAMETER CONFIGURATIONS")
    print("=" * 60)
    
    for exploration_type in ["conservative", "default", "extended"]:
        print(f"\n📋 Testing {exploration_type.upper()} configuration:")
        try:
            param_ranges = get_parameter_ranges("fvf", exploration_type)
            
            # Check that all required parameters are present
            required_params = [
                "device_type", "placement", 
                "width_1", "width_2", 
                "length_1", "length_2",
                "fingers_1", "fingers_2", 
                "multipliers_1", "multipliers_2", 
                "sd_rmult"
            ]
            
            missing_params = []
            for param in required_params:
                if param not in param_ranges:
                    missing_params.append(param)
            
            if missing_params:
                print(f"❌ Missing parameters: {missing_params}")
                return False
            else:
                print(f"✅ All {len(required_params)} parameters present")
                
            # Print summary
            print_parameter_summary(param_ranges)
            
        except Exception as e:
            print(f"❌ Error in {exploration_type}: {e}")
            return False
    
    return True

def test_parameter_sampling():
    """Test that parameter sampling works with new parameters"""
    print("\n🎲 TESTING PARAMETER SAMPLING")
    print("=" * 60)
    
    try:
        # Create explorer with conservative parameters for testing
        explorer = LHSParameterExplorer(database_path="test_db.pkl", use_extended_lhs=True)
        
        # Load parameter ranges
        param_ranges = get_parameter_ranges("fvf", "conservative")
        for param_range in param_ranges.values():
            explorer.add_parameter_range(param_range)
        
        print(f"✅ Added {len(param_ranges)} parameter ranges")
        
        # Generate a small sample to test
        print("Generating 5 test samples...")
        parameter_sets = explorer.generate_lhs_samples(5, seed=42, use_pruning=False)
        
        if len(parameter_sets) != 5:
            print(f"❌ Expected 5 samples, got {len(parameter_sets)}")
            return False
        
        print(f"✅ Generated {len(parameter_sets)} parameter combinations")
        
        # Check that all parameters are present in each sample
        for i, params in enumerate(parameter_sets):
            print(f"\nSample {i+1}:")
            for param_name, value in params.items():
                print(f"  {param_name:15s}: {value}")
            
            # Verify required parameters
            required_params = ["device_type", "placement", "width_1", "width_2"]
            for req_param in required_params:
                if req_param not in params:
                    print(f"❌ Missing parameter {req_param} in sample {i+1}")
                    return False
        
        print("✅ All samples contain required parameters")
        
        # Test parameter value validity
        print("\n🔍 Validating parameter values...")
        for i, params in enumerate(parameter_sets):
            # Check device_type
            if params['device_type'] not in ['nmos', 'pmos']:
                print(f"❌ Invalid device_type: {params['device_type']}")
                return False
            
            # Check placement
            if params['placement'] not in ['horizontal', 'vertical']:
                print(f"❌ Invalid placement: {params['placement']}")
                return False
            
            # Check numeric ranges
            if not (0.5 <= params['width_1'] <= 10.0):
                print(f"❌ width_1 out of range: {params['width_1']}")
                return False
        
        print("✅ All parameter values are valid")
        
        # Clean up test database
        if os.path.exists("test_db.pkl"):
            os.remove("test_db.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in parameter sampling test: {e}")
        traceback.print_exc()
        return False

def test_circuit_creation():
    """Test that FVF circuit can be created with new parameters"""
    print("\n🏗️  TESTING CIRCUIT CREATION")
    print("=" * 60)
    
    try:
        explorer = LHSParameterExplorer(database_path="test_db.pkl", use_extended_lhs=False)
        
        # Test different parameter combinations
        test_params = [
            {
                'device_type': 'nmos',
                'placement': 'horizontal',
                'width_1': 2.0, 'width_2': 1.5,
                'length_1': 0.18, 'length_2': 0.18,
                'fingers_1': 2, 'fingers_2': 1,
                'multipliers_1': 1, 'multipliers_2': 1,
                'sd_rmult': 2
            },
            {
                'device_type': 'pmos',
                'placement': 'vertical',
                'width_1': 3.0, 'width_2': 2.0,
                'length_1': 0.25, 'length_2': 0.20,
                'fingers_1': 1, 'fingers_2': 2,
                'multipliers_1': 2, 'multipliers_2': 1,
                'sd_rmult': 1
            }
        ]
        
        for i, params in enumerate(test_params):
            print(f"\nTesting parameter set {i+1}:")
            print(f"  Device: {params['device_type']}, Placement: {params['placement']}")
            
            try:
                circuit, instance_id = explorer.create_fvf_circuit(params)
                print(f"✅ Successfully created circuit: {circuit.name}")
                print(f"   Instance ID: {instance_id}")
                
                # Verify circuit has expected properties
                if hasattr(circuit, 'ports'):
                    print(f"   Ports: {len(circuit.ports)} ports available")
                
            except Exception as e:
                print(f"❌ Failed to create circuit: {e}")
                return False
        
        # Clean up
        if os.path.exists("test_db.pkl"):
            os.remove("test_db.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in circuit creation test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 FULL FVF PARAMETER SWEEP TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Parameter Configurations", test_parameter_configurations),
        ("Parameter Sampling", test_parameter_sampling),
        ("Circuit Creation", test_circuit_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ❌ FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print("="*80)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Parameter sweeping is fully functional.")
        print("\nYou can now run:")
        print("  python run_fvf_optimization.py --exploration conservative --samples 50")
        print("  python run_fvf_optimization.py --exploration default --samples 100")
        print("  python run_fvf_optimization.py --exploration extended --samples 200")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 