#!/bin/bash

# ==============================================================================
# KAN-TTS Experiment Runner
# ==============================================================================
# This script runs experiments with multiple random seeds for statistical significance.
# It handles configuration verification, logging, and result aggregation.
# ==============================================================================

set -e  # Exit on error

# Configuration
NUM_RUNS=5  # Number of runs per model for statistical significance
SEEDS=(42 123 456 789 101112)  # Fixed seeds for reproducibility
MODELS=("baseline_mlp" "baseline_transformer" "kan_fkf_full_features" "kan_ablation_phonemes_only")
BASE_OUTPUT_DIR="experiments"

# Verify Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

# Create analysis scripts directory if it doesn't exist
mkdir -p analysis_scripts

# Verify configurations before running experiments
echo "🔍 Verifying configuration consistency..."
python3 analysis_scripts/verify_configs.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration verification failed. Please fix inconsistencies before running experiments."
    exit 1
fi
echo "✅ Configuration verification passed"

# Function to run a single experiment
run_experiment() {
    local model=$1
    local seed=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="${BASE_OUTPUT_DIR}/${model}/seed_${seed}_${timestamp}"
    
    echo "🚀 Running experiment: ${model} (seed: ${seed})"
    echo "📁 Output directory: ${output_dir}"
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    # Copy configuration file to preserve exact settings
    cp "configs/${model}.yaml" "${output_dir}/config.yaml"
    
    # Save git commit hash
    git rev-parse HEAD > "${output_dir}/git_commit.txt" 2>/dev/null || echo "Not a git repository" > "${output_dir}/git_commit.txt"
    
    # Save experiment metadata
    cat > "${output_dir}/metadata.txt" << EOF
Experiment: ${model}
Seed: ${seed}
Timestamp: ${timestamp}
Date: $(date)
User: $(whoami)
Host: $(hostname)
Python Version: $(python3 --version)
PyTorch Version: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not available")
EOF
    
    # Run training with seed as argument
    PYTHONPATH=. python3 train.py \
        --config "configs/${model}.yaml" \
        --output_dir "${output_dir}" \
        --experiment_id "${model}_${seed}" \
        --seed "${seed}" \
        2>&1 | tee "${output_dir}/training.log"
    
    # Calculate and log metrics
    python3 analysis_scripts/calculate_metrics.py \
        --model_dir "${output_dir}" \
        --output_file "${output_dir}/metrics.json"
}

# Function to aggregate results across seeds
aggregate_results() {
    local model=$1
    echo "📊 Aggregating results for ${model}..."
    
    python3 analysis_scripts/aggregate_metrics.py \
        --base_dir "${BASE_OUTPUT_DIR}/${model}" \
        --output_file "${BASE_OUTPUT_DIR}/${model}/aggregated_metrics.json"
}

# Main execution
echo "🎯 Starting experiments with ${NUM_RUNS} runs per model"

# Create a pre-run verification report
echo "📋 Creating pre-run verification report..."
cat > "${BASE_OUTPUT_DIR}/pre_run_verification.txt" << EOF
PRE-EXPERIMENT VERIFICATION REPORT
================================
Date: $(date)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "Not a git repository")

Models to be tested:
${MODELS[@]}

Random Seeds:
${SEEDS[@]}

Configuration Status:
EOF

python3 analysis_scripts/verify_configs.py >> "${BASE_OUTPUT_DIR}/pre_run_verification.txt" 2>&1

# Run experiments
for model in "${MODELS[@]}"; do
    echo "📋 Processing model: ${model}"
    
    # Verify clean git state
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "✅ Git working directory clean"
    else
        echo "⚠️  Warning: Uncommitted changes in working directory"
        echo "Please commit your changes before running experiments"
        exit 1
    fi
    
    # Run experiments with different seeds
    for seed in "${SEEDS[@]}"; do
        run_experiment "${model}" "${seed}"
        
        # Quick validation after first epoch
        echo "⏳ Waiting for first epoch to complete..."
        sleep 60  # Adjust based on your epoch duration
        
        # Check if training is progressing
        if ! tail -n 50 "${output_dir}/training.log" | grep -q "train_loss"; then
            echo "❌ No training loss found in log. Please check the experiment."
            exit 1
        fi
    done
    
    # Aggregate results
    aggregate_results "${model}"
done

# Generate final comparison report
echo "📑 Generating final comparison report..."
python3 analysis_scripts/generate_report.py \
    --base_dir "${BASE_OUTPUT_DIR}" \
    --output_file "${BASE_OUTPUT_DIR}/final_comparison_report.pdf"

# Create experiment narrative template
cat > "${BASE_OUTPUT_DIR}/experiment_narrative.md" << EOF
# KAN-TTS Experiment Narrative

## Overview
Date: $(date)
Experimenter: $(whoami)

## Baseline Performance (MLP)
- Key Observations:
- Performance Analysis:
- Insights:

## Transformer Results
- Performance Comparison:
- Key Strengths/Weaknesses:
- Insights:

## KAN Model Analysis
- Performance vs Baselines:
- Feature Importance Findings:
- Interpretability Insights:

## Ablation Study Results
- Impact of Linguistic Features:
- Key Findings:
- Implications:

## Overall Narrative
- Main Story:
- Supporting Evidence:
- Challenges and Insights:

## Next Steps
- Areas for Investigation:
- Follow-up Experiments:
- Open Questions:
EOF

echo "✅ All experiments completed successfully"
echo "📊 Results available in: ${BASE_OUTPUT_DIR}/final_comparison_report.pdf"
echo "📝 Please document your observations in: ${BASE_OUTPUT_DIR}/experiment_narrative.md" 