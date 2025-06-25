# comprehensive evaluator
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Fix PDK_ROOT path to point to the correct Sky130 PDK installation
os.environ['PDK_ROOT'] = '/Applications/conda/miniconda3/envs/openfasoc-env/share/pdk'

from gdsfactory.typings import Component

from verification import run_verification
from physical_features import run_physical_feature_extraction

# Create centralized reports directory
REPORTS_DIR = Path("reports").resolve()
REPORTS_DIR.mkdir(exist_ok=True)

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

def setup_component_reports_dir(component_name: str) -> Path:
    """
    Creates a dedicated directory for component reports and returns the path.
    """
    component_reports_dir = REPORTS_DIR / component_name
    component_reports_dir.mkdir(exist_ok=True)
    return component_reports_dir

def cleanup_intermediate_files(component_name: str):
    """
    Deletes known intermediate and report files for a given component to ensure a clean run.
    Now also cleans up the centralized reports directory.
    """
    print(f"Cleaning up intermediate files for component '{component_name}'...")
    
    # Clean up files in current directory
    files_to_delete = [
        f"{component_name}.res.ext",
        f"{component_name}.lvs.rpt",
        f"{component_name}.drc.rpt",
        f"{component_name}.nodes",
        f"{component_name}.sim",
        f"{component_name}.pex.spice", 
        f"{component_name}_pex.spice",
        f"{component_name}_lvsmag.spice",
        f"{component_name}_sim.spice"
    ]
    
    for f_path in files_to_delete:
        try:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"  - Deleted: {f_path}")
        except OSError as e:
            print(f"  - Warning: Could not delete {f_path}. Error: {e}")
    
    # Clean up files in reports directory
    component_reports_dir = REPORTS_DIR / component_name
    if component_reports_dir.exists():
        for f_path in component_reports_dir.glob("*"):
            try:
                f_path.unlink()
                print(f"  - Deleted: {f_path}")
            except OSError as e:
                print(f"  - Warning: Could not delete {f_path}. Error: {e}")

def run_evaluation(layout_path: str, component_name: str, top_level: Component) -> dict:
    """
    The main evaluation wrapper. Runs all evaluation modules and combines results.
    Now uses centralized reports directory for all generated files.
    """
    print(f"--- Starting Comprehensive Evaluation for {component_name} ---")

    # Setup centralized reports directory for this component
    component_reports_dir = setup_component_reports_dir(component_name)
    print(f"Reports will be stored in: {component_reports_dir}")

    # Clean up previous runs
    cleanup_intermediate_files(component_name)

    # Run physical features module first so that PEX-derived netlists are available
    # for the LVS portion of the verification step.
    print("Running physical feature extraction (PEX, Area, Symmetry)...")
    physical_results = run_physical_feature_extraction(layout_path, component_name, top_level, component_reports_dir)

    # Run verification module with centralized reports (DRC, LVS)
    print("Running verification checks (DRC, LVS)...")
    verification_results = run_verification(layout_path, component_name, top_level, component_reports_dir)
    
    # Combine results into a single dictionary
    final_results = {
        "component_name": component_name,
        "timestamp": datetime.now().isoformat(),
        "reports_directory": str(component_reports_dir),
        "drc_lvs_fail": not (verification_results["drc"]["is_pass"] and verification_results["lvs"]["is_pass"]),
        **verification_results,
        **physical_results
    }
    
    # Log the final combined dictionary
    logger.info(final_results)
    print(f"--- Evaluation complete. Results logged to {log_file_name} ---")
    print(f"--- All reports stored in: {component_reports_dir} ---")
    
    return final_results

