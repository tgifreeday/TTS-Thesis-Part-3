import yaml
import sys
from typing import Dict, List, Any
import difflib
from pathlib import Path

def load_yaml(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def compare_configs(config1: Dict, config2: Dict, ignore_keys: List[str]) -> List[str]:
    """Compare two configurations, ignoring specified keys."""
    differences = []
    
    def compare_dict(d1: Dict, d2: Dict, path: str = ""):
        for key in set(d1.keys()) | set(d2.keys()):
            if key in ignore_keys:
                continue
                
            new_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                differences.append(f"Missing in config1: {new_path}")
            elif key not in d2:
                differences.append(f"Missing in config2: {new_path}")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dict(d1[key], d2[key], new_path)
            elif d1[key] != d2[key]:
                differences.append(f"Value mismatch at {new_path}:")
                differences.append(f"  Config1: {d1[key]}")
                differences.append(f"  Config2: {d2[key]}")
    
    compare_dict(config1, config2)
    return differences

def verify_experiment_configs(base_dir: str = "../configs"):
    """Verify that configurations are consistent across models."""
    # Keys that are expected to differ between models
    ignore_keys = [
        "run_name", "experiment_id", "description",
        "prosody_predictor", "architecture_type",
        "mlp", "transformer", "kan"  # Model-specific sections
    ]
    
    config_files = {
        "baseline_mlp": "baseline_mlp.yaml",
        "baseline_transformer": "baseline_transformer.yaml",
        "kan_fkf": "kan_fkf_full_features.yaml",
        "kan_ablation": "kan_ablation_phonemes_only.yaml"
    }
    
    configs = {}
    for model, filename in config_files.items():
        file_path = Path(base_dir) / filename
        if not file_path.exists():
            print(f"Error: Configuration file not found: {file_path}")
            continue
        configs[model] = load_yaml(str(file_path))
    
    # Compare each pair of configurations
    models = list(configs.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            print(f"\nComparing {model1} vs {model2}:")
            differences = compare_configs(configs[model1], configs[model2], ignore_keys)
            
            if differences:
                print("Found differences:")
                for diff in differences:
                    print(f"  {diff}")
            else:
                print("✓ Configurations are identical (excluding model-specific sections)")
    
    # Verify random seeds are properly handled
    for model, config in configs.items():
        seed_handling = config.get("reproducibility", {}).get("seed")
        if seed_handling is not None and seed_handling != "null":
            print(f"\nWarning: {model} has hardcoded seed: {seed_handling}")
            print("Seeds should be set to null and managed by run_experiment.sh")

if __name__ == "__main__":
    verify_experiment_configs() 