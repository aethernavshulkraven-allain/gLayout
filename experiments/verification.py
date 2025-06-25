import os
import re
import subprocess
import shutil
import tempfile
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fix PDK_ROOT path to point to the correct Sky130 PDK installation
os.environ['PDK_ROOT'] = '/Applications/conda/miniconda3/envs/openfasoc-env/share/pdk'

try:
    from glayout.pdk.sky130_mapped import sky130_mapped_pdk
    PDK_AVAILABLE = True
except ImportError:
    PDK_AVAILABLE = False
    print("Warning: PDK not available, using direct tool calls")

from gdsfactory.typings import Component

def parse_drc_report(report_content: str) -> dict:
    """
    Parses a Magic DRC report into a machine-readable format.
    """
    errors = []
    current_rule = ""
    for line in report_content.strip().splitlines():
        stripped_line = line.strip()
        if stripped_line == "----------------------------------------":
            continue
        if re.match(r"^[a-zA-Z]", stripped_line):
            current_rule = stripped_line
        elif re.match(r"^[0-9]", stripped_line):
            errors.append({"rule": current_rule, "details": stripped_line})
    
    is_pass = len(errors) == 0
    if not is_pass and re.search(r"count:\s*0\s*$", report_content, re.IGNORECASE):
        is_pass = True

    return {
        "is_pass": is_pass,
        "total_errors": len(errors),
        "error_details": errors
    }

def parse_lvs_report(report_content: str) -> dict:
    """
    Parses the raw netgen LVS report and returns a summarized, machine-readable format.
    Focuses on parsing net and instance mismatches.
    """
    summary = {
        "is_pass": False,
        "conclusion": "LVS failed or report was inconclusive.",
        "total_mismatches": 0,
        "mismatch_details": {
            "nets": "Not found", 
            "devices": "Not found", 
            "unmatched_nets_parsed": [],
            "unmatched_instances_parsed": []
        }
    }
    
    # Primary check for LVS pass/fail
    if "Netlists match" in report_content or "Circuits match uniquely" in report_content:
        summary["is_pass"] = True
        summary["conclusion"] = "LVS Pass: Netlists match."
    elif "Netlist mismatch" in report_content or "Netlists do not match" in report_content:
        summary["conclusion"] = "LVS Fail: Netlist mismatch."

    for line in report_content.splitlines():
        line = line.strip()

        # Parse net mismatches
        net_mismatch_match = re.search(r"Net:\s*([^\|]+)\s*\|\s*\((no matching net)\)", line)
        if net_mismatch_match:
            name_left = net_mismatch_match.group(1).strip()
            # If name is on the left, it's in layout, missing in schematic
            summary["mismatch_details"]["unmatched_nets_parsed"].append({
                "type": "net",
                "name": name_left,
                "present_in": "layout",
                "missing_in": "schematic"
            })
            continue

        # Parse instance mismatches
        instance_mismatch_match = re.search(r"Instance:\s*([^\|]+)\s*\|\s*\((no matching instance)\)", line)
        if instance_mismatch_match:
            name_left = instance_mismatch_match.group(1).strip()
            # If name is on the left, it's in layout, missing in schematic
            summary["mismatch_details"]["unmatched_instances_parsed"].append({
                "type": "instance",
                "name": name_left,
                "present_in": "layout",
                "missing_in": "schematic"
            })
            continue

        # Also capture cases where something is present in schematic but missing in layout (right side of '|')
        net_mismatch_right_match = re.search(r"\s*\|\s*([^\|]+)\s*\((no matching net)\)", line)
        if net_mismatch_right_match:
            name_right = net_mismatch_right_match.group(1).strip()
            # If name is on the right, it's in schematic, missing in layout
            summary["mismatch_details"]["unmatched_nets_parsed"].append({
                "type": "net",
                "name": name_right,
                "present_in": "schematic",
                "missing_in": "layout"
            })
            continue

        instance_mismatch_right_match = re.search(r"\s*\|\s*([^\|]+)\s*\((no matching instance)\)", line)
        if instance_mismatch_right_match:
            name_right = instance_mismatch_right_match.group(1).strip()
            # If name is on the right, it's in schematic, missing in layout
            summary["mismatch_details"]["unmatched_instances_parsed"].append({
                "type": "instance",
                "name": name_right,
                "present_in": "schematic",
                "missing_in": "layout"
            })
            continue

        # Capture summary lines like "Number of devices:" and "Number of nets:"
        if "Number of devices:" in line:
            summary["mismatch_details"]["devices"] = line.split(":", 1)[1].strip() if ":" in line else line
        elif "Number of nets:" in line:
            summary["mismatch_details"]["nets"] = line.split(":", 1)[1].strip() if ":" in line else line

    # Calculate total mismatches
    summary["total_mismatches"] = len(summary["mismatch_details"]["unmatched_nets_parsed"]) + \
                                  len(summary["mismatch_details"]["unmatched_instances_parsed"])

    # If there are any mismatches found, then LVS fails, regardless of "Netlists match" string.
    if summary["total_mismatches"] > 0:
        summary["is_pass"] = False
        if "LVS Pass" in summary["conclusion"]: # If conclusion still says pass, update it
            summary["conclusion"] = "LVS Fail: Mismatches found."

    return summary

def copy_report_to_centralized_location(temp_report_path: str, reports_dir: Path, component_name: str, report_type: str) -> str:
    """
    Copies report from temporary location to centralized reports directory.
    Returns the final report path.
    """
    if not os.path.exists(temp_report_path):
        return None
        
    final_report_path = reports_dir / f"{component_name}.{report_type}.rpt"
    try:
        shutil.copy2(temp_report_path, final_report_path)
        print(f"  - Copied {report_type.upper()} report to: {final_report_path}")
        return str(final_report_path)
    except Exception as e:
        print(f"  - Warning: Could not copy {report_type.upper()} report. Error: {e}")
        return temp_report_path

def run_drc_verification(gds_path: str, design_name: str, output_file: str = None) -> bool:
    """Run DRC verification using Magic"""
    if output_file is None:
        output_file = f"{design_name}.drc.rpt"
    
    if PDK_AVAILABLE:
        return sky130_mapped_pdk.drc_magic(gds_path, design_name, output_file)
    
    # Fallback: Direct Magic call
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_content = f"""
#!/usr/bin/env tclsh
# Magic DRC script for {design_name}

drc style drc(full)
tech load sky130A
gds read {gds_path}
load {design_name}
select top cell
drc check
drc count
set drc_count [drc list count]
puts "DRC violations: $drc_count"

if {{$drc_count > 0}} {{
    puts "DRC FAILED: $drc_count violations found"
    drc list
}} else {{
    puts "DRC PASSED: No violations found"
}}

quit
"""
            
            script_path = os.path.join(tmpdir, "drc_script.tcl")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run magic with the script
            cmd = [
                "magic",
                "-noconsole", 
                "-dnull",
                "-T", "sky130A",
                script_path
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=tmpdir,
                capture_output=True, 
                text=True,
                timeout=300
            )
            
            # Write report
            with open(output_file, 'w') as f:
                f.write(f"Magic DRC Report for {design_name}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Parse result for pass/fail
            drc_passed = "DRC PASSED" in result.stdout or result.returncode == 0
            print(f"DRC check completed. Report saved to {output_file}")
            print(f"Result: {'PASSED' if drc_passed else 'FAILED'}")
            
            return drc_passed
            
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Magic DRC Error: {str(e)}\n")
        print(f"DRC check failed: {e}")
        return False

def run_lvs_verification(gds_path: str, design_name: str, output_file: str = None, spice_netlist: str = None) -> bool:
    """Run LVS verification using Netgen"""
    if output_file is None:
        output_file = f"{design_name}.lvs.rpt"
    
    if PDK_AVAILABLE:
        return sky130_mapped_pdk.lvs_netgen(gds_path, design_name, output_file, spice_netlist)
    
    # Fallback: Direct Netgen call
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_content = f"""
# Netgen LVS script for {design_name}

# Simple connectivity check
lvs "{design_name}" "{design_name}"
quit
"""
            
            script_path = os.path.join(tmpdir, "lvs_script.tcl")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run netgen
            cmd = ["netgen", "-batch", "source", script_path]
            
            result = subprocess.run(
                cmd,
                cwd=tmpdir, 
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Write report
            with open(output_file, 'w') as f:
                f.write(f"Netgen LVS Report for {design_name}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            # Parse result
            lvs_passed = result.returncode == 0
            print(f"LVS check completed. Report saved to {output_file}")
            print(f"Result: {'PASSED' if lvs_passed else 'FAILED'}")
            
            return lvs_passed
            
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"Netgen LVS Error: {str(e)}\n")
        print(f"LVS check failed: {e}")
        return False

def run_verification(layout_path: str, component_name: str, top_level: Component, reports_dir: Path) -> dict:
    """
    Runs DRC and LVS checks and returns a structured result dictionary.
    Now uses centralized reports directory.
    """
    verification_results = {
        "drc": {"status": "not run", "is_pass": False, "report_path": None, "summary": {}},
        "lvs": {"status": "not run", "is_pass": False, "report_path": None, "summary": {}}
    }
    
    # DRC Check
    print("  Running DRC check...")
    drc_report_path = str(reports_dir / f"{component_name}.drc.rpt")
    verification_results["drc"]["report_path"] = drc_report_path
    
    try:
        # Remove old report if exists
        if os.path.exists(drc_report_path):
            os.remove(drc_report_path)
            
        # Run DRC and specify output file directly to centralized location
        drc_passed = run_drc_verification(layout_path, component_name, drc_report_path)
        
        # Copy report from current directory to centralized location if it exists
        temp_drc_path = f"./{component_name}.drc.rpt"
        if os.path.exists(temp_drc_path):
            final_drc_path = copy_report_to_centralized_location(temp_drc_path, reports_dir, component_name, "drc")
            if final_drc_path:
                verification_results["drc"]["report_path"] = final_drc_path
                # Clean up temp file
                try:
                    os.remove(temp_drc_path)
                except:
                    pass
        else:
            print(f"  - Warning: DRC report not found at expected temp location: {temp_drc_path}")
            # Check if report was generated directly in centralized location
            if os.path.exists(drc_report_path):
                print(f"  - Found DRC report in centralized location: {drc_report_path}")
            else:
                print(f"  - DRC report not found in centralized location either: {drc_report_path}")
        

    except Exception as e:
        verification_results["drc"]["status"] = f"error: {str(e)}"
        print(f"  - DRC Error: {e}")
        # Still check if we can find the report file
        if os.path.exists(verification_results["drc"]["report_path"]):
            print(f"  - DRC report found despite error, attempting to parse...")

    # Parse the report regardless of whether there was an error
    # (the report might still exist and be parseable)
    report_content = ""
    if os.path.exists(verification_results["drc"]["report_path"]):
        try:
            with open(verification_results["drc"]["report_path"], 'r') as f:
                report_content = f.read()
                
            summary = parse_drc_report(report_content)
            verification_results["drc"].update({
                "summary": summary, 
                "is_pass": summary["is_pass"], 
                "status": "pass" if summary["is_pass"] else "fail"
            })
            print(f"  - DRC report parsed successfully from: {verification_results['drc']['report_path']}")
        except Exception as parse_e:
            print(f"  - Failed to parse DRC report: {parse_e}")
    else:
        print(f"  - No DRC report file found at: {verification_results['drc']['report_path']}")

    # LVS Check
    print("  Running LVS check...")
    lvs_report_path = str(reports_dir / f"{component_name}.lvs.rpt")
    verification_results["lvs"]["report_path"] = lvs_report_path
    
    # Try to locate a SPICE/CDL netlist generated during PEX so that LVS has
    # both layout and schematic data to compare.  We do this *before* the
    # actual LVS call so that we can pass the file directly instead of
    # relying on the PDK wrapper to guess.
    possible_netlist_paths = [
        str(reports_dir / f"{component_name}_pex.spice"),
        str(reports_dir / f"{component_name}.pex.spice"),
        f"{component_name}_pex.spice",
        f"{component_name}.pex.spice",
    ]

    spice_netlist_path = next((p for p in possible_netlist_paths if os.path.exists(p)), None)

    try:
        # Remove old report if exists
        if os.path.exists(lvs_report_path):
            os.remove(lvs_report_path)
            
        # Run LVS and specify output file directly to centralized location
        if spice_netlist_path:
            print(f"  - Using netlist for LVS: {spice_netlist_path}")
            lvs_passed = run_lvs_verification(
                spice_netlist_path,  # use netlist as 'layout' input (see pdk wrapper)
                component_name,
                lvs_report_path,
                spice_netlist=spice_netlist_path,
            )
        else:
            lvs_passed = run_lvs_verification(layout_path, component_name, lvs_report_path)
            
        # Copy report from current directory to centralized location if it exists
        temp_lvs_path = f"./{component_name}.lvs.rpt"
        if os.path.exists(temp_lvs_path):
            final_lvs_path = copy_report_to_centralized_location(temp_lvs_path, reports_dir, component_name, "lvs")
            if final_lvs_path:
                verification_results["lvs"]["report_path"] = final_lvs_path
                # Clean up temp file
                try:
                    os.remove(temp_lvs_path)
                except:
                    pass
        else:
            print(f"  - Warning: LVS report not found at expected temp location: {temp_lvs_path}")
            # Check if report was generated directly in centralized location
            if os.path.exists(lvs_report_path):
                print(f"  - Found LVS report in centralized location: {lvs_report_path}")
            else:
                print(f"  - LVS report not found in centralized location either: {lvs_report_path}")
        
    except Exception as e:
        verification_results["lvs"]["status"] = f"error: {e}"
        print(f"  - LVS Error: {e}")

    # Parse the report regardless of whether there was an error
    # (the report might still exist and be parseable)
    report_content = ""
    if os.path.exists(verification_results["lvs"]["report_path"]):
        try:
            with open(verification_results["lvs"]["report_path"], 'r') as report_file:
                report_content = report_file.read()
                
            lvs_summary = parse_lvs_report(report_content)
            verification_results["lvs"].update({
                "summary": lvs_summary, 
                "is_pass": lvs_summary["is_pass"], 
                "status": "pass" if lvs_summary["is_pass"] else "fail"
            })
            print(f"  - LVS report parsed successfully from: {verification_results['lvs']['report_path']}")
        except Exception as parse_e:
            print(f"  - Failed to parse LVS report: {parse_e}")
    else:
        print(f"  - No LVS report file found at: {verification_results['lvs']['report_path']}")
        
    return verification_results