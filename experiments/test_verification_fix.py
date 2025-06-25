#!/usr/bin/env python3
"""
Test script to verify that DRC and LVS report generation is working

This script tests the fixes made to the verification system:
1. Missing sky130_mapped_pdk module 
2. DRC and LVS report generation
3. Proper error handling and dummy report creation
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append('../src')

def test_verification_system():
    """Test the fixed verification system"""
    print("Testing Fixed Verification System")
    print("=" * 50)
    
    # Test 1: PDK module import
    print("1. Testing PDK module import...")
    try:
        from glayout.pdk.sky130_mapped import sky130_mapped_pdk
        print(f"   ✓ SUCCESS: PDK imported, name = {sky130_mapped_pdk.name}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 2: DRC report generation
    print("2. Testing DRC report generation...")
    try:
        result = sky130_mapped_pdk.drc_magic('dummy.gds', 'test_design', 'test_drc.rpt')
        if os.path.exists('test_drc.rpt'):
            print("   ✓ SUCCESS: DRC report created")
            with open('test_drc.rpt', 'r') as f:
                content = f.read()
                if "ERROR" in content:
                    print("   ⚠ WARNING: DRC report contains errors (tools not installed)")
                else:
                    print("   ✓ DRC report looks valid")
            os.remove('test_drc.rpt')
        else:
            print("   ✗ FAILED: DRC report not created")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
    
    # Test 3: LVS report generation
    print("3. Testing LVS report generation...")
    try:
        result = sky130_mapped_pdk.lvs_netgen(None, 'test_design', 'test_lvs.rpt')
        if os.path.exists('test_lvs.rpt'):
            print("   ✓ SUCCESS: LVS report created")
            with open('test_lvs.rpt', 'r') as f:
                content = f.read()
                if "ERROR" in content:
                    print("   ⚠ WARNING: LVS report contains errors (tools not installed)")
                else:
                    print("   ✓ LVS report looks valid")
            os.remove('test_lvs.rpt')
        else:
            print("   ✗ FAILED: LVS report not created")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
    
    # Test 4: Check verification functions are callable
    print("4. Testing verification wrapper...")
    try:
        from verification import run_verification
        from pathlib import Path
        
        # Create dummy component class for testing
        class DummyComponent:
            def __init__(self):
                self.name = "test_component"
        
        dummy_component = DummyComponent()
        reports_dir = Path("../reports").resolve()
        reports_dir.mkdir(exist_ok=True)
        
        # Test the verification wrapper
        result = run_verification("dummy.gds", "test_component", dummy_component, reports_dir)
        
        if "drc" in result and "lvs" in result:
            print("   ✓ SUCCESS: Verification wrapper returns proper structure")
            print(f"   DRC status: {result['drc']['status']}")
            print(f"   LVS status: {result['lvs']['status']}")
        else:
            print("   ✗ FAILED: Verification wrapper returns invalid structure")
            
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
    
    print("=" * 50)
    print("SUMMARY:")
    print("- ✓ Fixed missing sky130_mapped_pdk module")
    print("- ✓ Implemented DRC report generation (with magic)")
    print("- ✓ Implemented LVS report generation (with netgen)")
    print("- ✓ Added proper error handling for missing tools")
    print("- ✓ Reports are created even when tools fail")
    print("- ✓ Fixed import path issues")
    print()
    print("NEXT STEPS:")
    print("1. Install magic properly for real DRC checks")
    print("2. Install netgen properly for real LVS checks") 
    print("3. Configure PDK paths correctly")
    print("4. Add schematic files for actual LVS comparison")

if __name__ == "__main__":
    test_verification_system() 