"""
Parameter configuration file for circuit exploration.
Modify these ranges based on your inventory and requirements.
"""

from typing import Dict, List
from lhs_parameter_explorer import ParameterRange

# FVF Parameter Configurations
# These can be modified based on your inventory Excel file

FVF_PARAMETER_RANGES = {
    # Device type: [nmos, pmos]
    "device_type": ParameterRange("device_type", min_val=0, max_val=1, param_type="discrete", 
                                 discrete_values=["nmos", "pmos"]),
    
    # Placement: [horizontal, vertical]
    "placement": ParameterRange("placement", min_val=0, max_val=1, param_type="discrete",
                               discrete_values=["horizontal", "vertical"]),
    
    # Transistor widths (in micrometers): [0.5, 10, 0.25] x 2
    "width_1": ParameterRange("width_1", min_val=0.5, max_val=10.0, param_type="continuous"),
    "width_2": ParameterRange("width_2", min_val=0.5, max_val=10.0, param_type="continuous"),
    
    # Transistor lengths (in micrometers): [0.15, 4, 0.2] x 2
    "length_1": ParameterRange("length_1", min_val=0.15, max_val=4.0, param_type="continuous"),
    "length_2": ParameterRange("length_2", min_val=0.15, max_val=4.0, param_type="continuous"),
    
    # Number of fingers (discrete values): [1, 5, 1] x 2
    "fingers_1": ParameterRange("fingers_1", min_val=1, max_val=5, param_type="discrete", 
                               discrete_values=[1, 2, 3, 4, 5]),
    "fingers_2": ParameterRange("fingers_2", min_val=1, max_val=5, param_type="discrete", 
                               discrete_values=[1, 2, 3, 4, 5]),
    
    # Number of multipliers (discrete values): [1, 2, 1] x 2
    "multipliers_1": ParameterRange("multipliers_1", min_val=1, max_val=2, param_type="discrete", 
                                   discrete_values=[1, 2]),
    "multipliers_2": ParameterRange("multipliers_2", min_val=1, max_val=2, param_type="discrete", 
                                   discrete_values=[1, 2]),
    
    # Source/drain routing multiplier
    "sd_rmult": ParameterRange("sd_rmult", min_val=1, max_val=5, param_type="discrete", 
                              discrete_values=[1, 2, 3, 4, 5]),
}

# Conservative parameter ranges (smaller search space for initial exploration)
FVF_PARAMETER_RANGES_CONSERVATIVE = {
    # Device type: [nmos, pmos]
    "device_type": ParameterRange("device_type", min_val=0, max_val=1, param_type="discrete", 
                                 discrete_values=["nmos", "pmos"]),
    
    # Placement: [horizontal, vertical]
    "placement": ParameterRange("placement", min_val=0, max_val=1, param_type="discrete",
                               discrete_values=["horizontal", "vertical"]),
    
    # Conservative width ranges
    "width_1": ParameterRange("width_1", min_val=1.0, max_val=6.0, param_type="continuous"),
    "width_2": ParameterRange("width_2", min_val=1.0, max_val=4.0, param_type="continuous"),
    
    # Conservative length ranges
    "length_1": ParameterRange("length_1", min_val=0.18, max_val=2.0, param_type="continuous"),
    "length_2": ParameterRange("length_2", min_val=0.18, max_val=2.0, param_type="continuous"),
    
    # Conservative fingers
    "fingers_1": ParameterRange("fingers_1", min_val=1, max_val=4, param_type="discrete", 
                               discrete_values=[1, 2, 3, 4]),
    "fingers_2": ParameterRange("fingers_2", min_val=1, max_val=4, param_type="discrete", 
                               discrete_values=[1, 2, 3, 4]),
    
    # Conservative multipliers
    "multipliers_1": ParameterRange("multipliers_1", min_val=1, max_val=2, param_type="discrete", 
                                   discrete_values=[1, 2]),
    "multipliers_2": ParameterRange("multipliers_2", min_val=1, max_val=2, param_type="discrete", 
                                   discrete_values=[1, 2]),
    
    # Conservative sd_rmult
    "sd_rmult": ParameterRange("sd_rmult", min_val=1, max_val=3, param_type="discrete", 
                              discrete_values=[1, 2, 3]),
}

# Extended parameter ranges (larger search space)
FVF_PARAMETER_RANGES_EXTENDED = {
    # Device type: [nmos, pmos]
    "device_type": ParameterRange("device_type", min_val=0, max_val=1, param_type="discrete", 
                                 discrete_values=["nmos", "pmos"]),
    
    # Placement: [horizontal, vertical]
    "placement": ParameterRange("placement", min_val=0, max_val=1, param_type="discrete",
                               discrete_values=["horizontal", "vertical"]),
    
    # Extended width ranges
    "width_1": ParameterRange("width_1", min_val=0.5, max_val=15.0, param_type="continuous"),
    "width_2": ParameterRange("width_2", min_val=0.5, max_val=15.0, param_type="continuous"),
    
    # Extended length ranges
    "length_1": ParameterRange("length_1", min_val=0.15, max_val=5.0, param_type="continuous"),
    "length_2": ParameterRange("length_2", min_val=0.15, max_val=5.0, param_type="continuous"),
    
    # Extended fingers
    "fingers_1": ParameterRange("fingers_1", min_val=1, max_val=12, param_type="discrete", 
                               discrete_values=list(range(1, 13))),
    "fingers_2": ParameterRange("fingers_2", min_val=1, max_val=12, param_type="discrete", 
                               discrete_values=list(range(1, 13))),
    
    # Extended multipliers
    "multipliers_1": ParameterRange("multipliers_1", min_val=1, max_val=6, param_type="discrete", 
                                   discrete_values=list(range(1, 7))),
    "multipliers_2": ParameterRange("multipliers_2", min_val=1, max_val=6, param_type="discrete", 
                                   discrete_values=list(range(1, 7))),
    
    # Extended sd_rmult
    "sd_rmult": ParameterRange("sd_rmult", min_val=1, max_val=8, param_type="discrete", 
                              discrete_values=list(range(1, 9))),
}

# Future: OpAmp parameter ranges (template for when you move to OpAmp)
OPAMP_PARAMETER_RANGES_TEMPLATE = {
    # Differential pair
    "diff_width": ParameterRange("diff_width", min_val=1.0, max_val=10.0, param_type="continuous"),
    "diff_length": ParameterRange("diff_length", min_val=0.15, max_val=2.0, param_type="continuous"),
    "diff_fingers": ParameterRange("diff_fingers", min_val=1, max_val=8, param_type="discrete", 
                                  discrete_values=list(range(1, 9))),
    
    # Current mirror load
    "load_width": ParameterRange("load_width", min_val=1.0, max_val=15.0, param_type="continuous"),
    "load_length": ParameterRange("load_length", min_val=0.15, max_val=2.0, param_type="continuous"),
    "load_fingers": ParameterRange("load_fingers", min_val=1, max_val=8, param_type="discrete", 
                                  discrete_values=list(range(1, 9))),
    
    # Output stage
    "output_width": ParameterRange("output_width", min_val=5.0, max_val=50.0, param_type="continuous"),
    "output_length": ParameterRange("output_length", min_val=0.15, max_val=2.0, param_type="continuous"),
    
    # Bias current (could be represented as current mirror ratios)
    "bias_ratio": ParameterRange("bias_ratio", min_val=0.5, max_val=5.0, param_type="continuous"),
}

def get_parameter_ranges(circuit_type: str = "fvf", exploration_type: str = "default") -> Dict[str, ParameterRange]:
    """
    Get parameter ranges for a specific circuit type and exploration level.
    
    Args:
        circuit_type: "fvf" or "opamp"
        exploration_type: "conservative", "default", or "extended"
    
    Returns:
        Dictionary of parameter ranges
    """
    if circuit_type.lower() == "fvf":
        if exploration_type == "conservative":
            return FVF_PARAMETER_RANGES_CONSERVATIVE
        elif exploration_type == "extended":
            return FVF_PARAMETER_RANGES_EXTENDED
        else:
            return FVF_PARAMETER_RANGES
    
    elif circuit_type.lower() == "opamp":
        return OPAMP_PARAMETER_RANGES_TEMPLATE
    
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

def print_parameter_summary(param_ranges: Dict[str, ParameterRange]):
    """Print a summary of parameter ranges"""
    print("Parameter Ranges Summary:")
    print("=" * 60)
    
    discrete_combinations = 1
    continuous_params = 0
    
    for name, param in param_ranges.items():
        if param.param_type == "continuous":
            print(f"{name:15s}: {param.min_val:6.2f} - {param.max_val:6.2f} (continuous)")
            continuous_params += 1
        elif param.param_type == "discrete":
            if param.discrete_values:
                print(f"{name:15s}: {param.discrete_values} (discrete)")
                discrete_combinations *= len(param.discrete_values)
            else:
                print(f"{name:15s}: {int(param.min_val)} - {int(param.max_val)} (discrete)")
                discrete_combinations *= int(param.max_val - param.min_val + 1)
    
    print("=" * 60)
    print(f"Discrete parameter combinations: {discrete_combinations:,}")
    print(f"Continuous parameters: {continuous_params}")
    print(f"Total parameter space: {discrete_combinations:,} discrete × continuous space")
    
    # Calculate key parameter space insights
    if 'device_type' in param_ranges and 'placement' in param_ranges:
        print(f"Device types × Placements: {len(param_ranges['device_type'].discrete_values)} × {len(param_ranges['placement'].discrete_values)} = {len(param_ranges['device_type'].discrete_values) * len(param_ranges['placement'].discrete_values)} base combinations") 