# comprehensive evaluator
import os
import json
import logging
from datetime import datetime
from gdsfactory.typings import Component

from verification import run_verification
from physical_features import run_physical_feature_extraction

def get_next_log_filename(base_name="evaluation", extension=".log"):
    """Generates the next available log filename with a numerical suffix."""
    filename = f"{base_name}{extension}"
    if not os.path.exists(filename): return filename
    i = 1
    while True:
        filename = f"{base_name}_{i}{extension}"
        if not os.path.exists(filename): return filename
        i += 1

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.msg) if isinstance(record.msg, dict) else super().format(record)

# Setup logger
log_file_name = get_next_log_filename()
log_file_handler = logging.FileHandler(log_file_name)
log_file_handler.setFormatter(JsonFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(log_file_handler)
logger.setLevel(logging.INFO)

def run_evaluation(layout_path: str, component_name: str, top_level: Component) -> dict:
    """
    The main evaluation wrapper. Runs all evaluation modules and combines results.
    """
    print(f"--- Starting Comprehensive Evaluation for {component_name} ---")

    #Deletes known intermediate and report files for a given component to ensure a clean run.
    print(f"Cleaning up intermediate files for component '{component_name}'...")
    
    files_to_delete = [
        f"{component_name}.res.ext",
        f"{component_name}.lvs.rpt",
        f"{component_name}.nodes",
        f"{component_name}.sim",
        f"{component_name}.pex.spice", 
        f"{component_name}_pex.spice" 
    ]
    
    for f_path in files_to_delete:
        try:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"  - Deleted: {f_path}")
        except OSError as e:
            print(f"  - Warning: Could not delete {f_path}. Error: {e}")

    # Run verification module
    print("Running verification checks (DRC, LVS)...")
    verification_results = run_verification(layout_path, component_name, top_level)
    
    # Run physical features module
    print("Running physical feature extraction (PEX, Area, Symmetry)...")
    physical_results = run_physical_feature_extraction(layout_path, component_name, top_level)
    
    # Combine results into a single dictionary
    final_results = {
        "component_name": component_name,
        "timestamp": datetime.now().isoformat(),
        "drc_lvs_fail": not (verification_results["drc"]["is_pass"] and verification_results["lvs"]["is_pass"]),
        **verification_results,
        **physical_results
    }
    
    # Log the final combined dictionary
    logger.info(final_results)
    print(f"--- Evaluation complete. Results logged to {log_file_name} ---")
    
    return final_results

