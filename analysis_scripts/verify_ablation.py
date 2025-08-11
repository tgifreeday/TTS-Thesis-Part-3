import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Set, Any
import json

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def verify_feature_sets(full_config: Dict, ablation_config: Dict) -> List[str]:
    """Verify feature sets between full and ablation configs."""
    issues = []
    
    full_features = set(full_config['data']['input_features'])
    ablation_features = set(ablation_config['data']['input_features'])
    
    # Check that ablation only contains phoneme_id
    if ablation_features != {'phoneme_id'}:
        issues.append("❌ Ablation config should only contain 'phoneme_id' feature")
        issues.append(f"   Found: {ablation_features}")
    
    # Check that full features contain all necessary Booij features
    required_features = {
        'phoneme_id',
        'stress_level',
        'primary_stress_pos',
        'secondary_stress_pos',
        'pos_tag',
        'is_content_word',
        'word_position_in_sentence',
        'is_sentence_initial',
        'is_sentence_final',
        'syllable_count',
        'word_length',
        'is_compound',
        'is_loanword',
        'vowel_count',
        'consonant_count'
    }
    
    missing_features = required_features - full_features
    if missing_features:
        issues.append("❌ Full config missing required Booij features:")
        issues.append(f"   Missing: {missing_features}")
    
    return issues

def verify_model_architecture(full_config: Dict, ablation_config: Dict) -> List[str]:
    """Verify that model architectures are identical except for input dimension."""
    issues = []
    
    # Keys that should be identical
    check_keys = [
        'model.architecture_type',
        'model.prosody_predictor.name',
        'model.prosody_predictor.hidden_dims',
        'model.prosody_predictor.dropout',
        'model.prosody_predictor.activation',
        'model.prosody_predictor.batch_norm',
        'model.phoneme_encoder',
        'model.spectrogram_decoder'
    ]
    
    def get_nested_value(config: Dict, key_path: str) -> Any:
        keys = key_path.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
        return value
    
    for key in check_keys:
        full_value = get_nested_value(full_config, key)
        ablation_value = get_nested_value(ablation_config, key)
        
        if full_value != ablation_value:
            issues.append(f"❌ Architecture mismatch at {key}:")
            issues.append(f"   Full config: {full_value}")
            issues.append(f"   Ablation config: {ablation_value}")
    
    # Verify input dimension adjustment
    full_input_dim = full_config['model']['prosody_predictor']['input_dim']
    ablation_input_dim = ablation_config['model']['prosody_predictor']['input_dim']
    
    if ablation_input_dim != 1:  # Only phoneme_id
        issues.append("❌ Ablation input dimension should be 1 (phoneme_id only)")
        issues.append(f"   Found: {ablation_input_dim}")
    
    return issues

def verify_training_parameters(full_config: Dict, ablation_config: Dict) -> List[str]:
    """Verify that training parameters are identical."""
    issues = []
    
    # Parameters that must be identical
    check_params = [
        'training.optimizer',
        'training.learning_rate',
        'training.weight_decay',
        'training.scheduler',
        'training.max_epochs',
        'training.early_stopping_patience',
        'training.gradient_clip_val',
        'data.batch_size',
        'data.num_workers'
    ]
    
    def get_nested_value(config: Dict, param_path: str) -> Any:
        keys = param_path.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
        return value
    
    for param in check_params:
        full_value = get_nested_value(full_config, param)
        ablation_value = get_nested_value(ablation_config, param)
        
        if full_value != ablation_value:
            issues.append(f"❌ Training parameter mismatch at {param}:")
            issues.append(f"   Full config: {full_value}")
            issues.append(f"   Ablation config: {ablation_value}")
    
    return issues

def verify_data_pipeline(config_path: str) -> List[str]:
    """Verify that the data pipeline correctly handles feature selection."""
    issues = []
    
    try:
        # Import locally to avoid dependencies when just checking configs
        from src.dataset import TTSDataset
        
        config = load_config(config_path)
        dataset = TTSDataset(config['data'])
        
        # Check first batch
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
        
        # Verify input dimensions
        expected_features = len(config['data']['input_features'])
        actual_features = batch['input_features'].shape[-1]
        
        if actual_features != expected_features:
            issues.append("❌ Data pipeline dimension mismatch:")
            issues.append(f"   Expected features: {expected_features}")
            issues.append(f"   Actual features: {actual_features}")
        
    except Exception as e:
        issues.append("❌ Error testing data pipeline:")
        issues.append(f"   {str(e)}")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description='Verify ablation study setup')
    parser.add_argument('--full-config', type=str, required=True,
                      help='Path to full feature configuration')
    parser.add_argument('--ablation-config', type=str, required=True,
                      help='Path to ablation configuration')
    args = parser.parse_args()
    
    # Load configurations
    full_config = load_config(args.full_config)
    ablation_config = load_config(args.ablation_config)
    
    all_issues = []
    
    # Run verifications
    print("🔍 Verifying ablation study setup...")
    
    print("\n1. Checking feature sets...")
    issues = verify_feature_sets(full_config, ablation_config)
    all_issues.extend(issues)
    if not issues:
        print("✅ Feature sets correctly configured")
    
    print("\n2. Checking model architecture...")
    issues = verify_model_architecture(full_config, ablation_config)
    all_issues.extend(issues)
    if not issues:
        print("✅ Model architectures properly aligned")
    
    print("\n3. Checking training parameters...")
    issues = verify_training_parameters(full_config, ablation_config)
    all_issues.extend(issues)
    if not issues:
        print("✅ Training parameters identical")
    
    print("\n4. Testing data pipeline...")
    issues = verify_data_pipeline(args.ablation_config)
    all_issues.extend(issues)
    if not issues:
        print("✅ Data pipeline handling features correctly")
    
    # Summary
    if all_issues:
        print("\n❌ Found issues that need to be fixed:")
        for issue in all_issues:
            print(issue)
        exit(1)
    else:
        print("\n✅ All verifications passed! Ablation study setup is correct.")
        print("You can proceed with training both models.")

if __name__ == '__main__':
    main() 