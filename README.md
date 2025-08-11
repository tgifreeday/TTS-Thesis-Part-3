# KAN-TTS Thesis: Kolmogorov-Arnold Networks for Dutch Text-to-Speech Synthesis

## 📋 Project Overview

This repository contains the implementation and experimental framework for the thesis research on **Kolmogorov-Arnold Networks (KANs) for Dutch Text-to-Speech Synthesis with Booij-Validated Linguistic Features**.

### 🎯 Research Objectives

**Primary Hypothesis (H1)**: KAN-based prosody predictors, informed by Booij-validated linguistic features, outperform traditional architectures for Dutch TTS synthesis.

**Secondary Hypotheses**:
- **H2**: Booij stress features provide measurable improvement over phoneme-only approaches
- **H3**: KANs benefit more from linguistic features than transformer architectures
- **H4**: 30.9 hours of Dutch speech data is sufficient for robust KAN TTS training

### 🔬 Research Context

This work addresses the **English-centric bias** in TTS research by developing a Dutch-specific synthesis system. The research integrates **Geert Booij's phonological framework** ("The Phonology of Dutch") with **Kolmogorov-Arnold Networks** to create linguistically-informed, interpretable TTS models.

## 🏗️ Project Structure

```
KAN-TTS-Thesis/
├── 📜 README.md                    # This file
├── 📦 requirements.txt             # Python dependencies
├── 🚀 run_experiment.sh            # Single command to launch experiments
├── 📋 TODO.md                      # Step-by-step implementation plan
│
├── 📂 data/                        # Data management
│   ├── raw/                       # Original audio and transcripts (immutable)
│   └── processed/                 # Preprocessed data for training
│       ├── linguistic_features.json  # Final dataset with Booij features
│       ├── train_files.txt        # Training file list
│       └── val_files.txt          # Validation file list
│
├── ⚙️ configs/                     # Experiment configurations
│   ├── baseline_mlp.yaml          # MLP baseline experiment
│   ├── baseline_transformer.yaml  # Transformer baseline experiment
│   ├── kan_fkf_full_features.yaml # Primary KAN hypothesis test
│   ├── kan_ablation_phonemes_only.yaml # Feature ablation study
│   ├── kan_karn_duration.yaml     # Temporal KAN for duration
│   └── kan_convkan_decoder.yaml   # Convolutional KAN for spectrograms
│
├── 🧪 experiments/                 # All experimental outputs
│   └── YYYYMMDD_HHMM_experiment_name/  # Timestamped experiment directories
│       ├── checkpoints/           # Model weights (.pth files)
│       ├── samples/               # Generated audio samples
│       ├── logs/                  # TensorBoard logs
│       └── config.yaml            # Exact config used for this run
│
├── src/                           # Source code
│   ├── train.py                   # Main training script
│   ├── dataset.py                 # PyTorch Dataset implementation
│   ├── models/                    # Neural network architectures
│   │   ├── tts_model.py           # Main TTS model orchestrator
│   │   ├── phoneme_encoder.py     # Phoneme encoding component
│   │   ├── prosody_predictor.py   # Modular prosody predictors
│   │   └── spectrogram_decoder.py # Spectrogram generation
│   └── components/                # Reusable KAN components
│       └── kan_blocks.py          # KAN building blocks
│
├── 🔬 analysis/                   # Post-experiment analysis
│   ├── visualize.py               # B-spline and attention visualization
│   ├── compare_pipelines.py       # Objective metric comparison
│   └── interpretability.ipynb     # Jupyter notebook for analysis
│
└── 📖 thesis/                     # Thesis documentation
    ├── figures/                   # Plots and diagrams
    ├── tables/                    # Result tables
    └── logbooks/                  # Research logbooks and documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 32GB+ RAM
- 100GB+ storage space

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd KAN-TTS-Thesis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**:
   ```bash
   # Copy your preprocessed data to data/processed/
   cp /path/to/your/linguistic_features.json data/processed/
   cp /path/to/your/train_files.txt data/processed/
   cp /path/to/your/val_files.txt data/processed/
   ```

### Running Experiments

**Single Command Execution**:
```bash
# Run MLP baseline
bash run_experiment.sh baseline_mlp

# Run Transformer baseline  
bash run_experiment.sh baseline_transformer

# Run primary KAN hypothesis test
bash run_experiment.sh kan_fkf_full_features

# Run feature ablation study
bash run_experiment.sh kan_ablation_phonemes_only
```

**Manual Training**:
```bash
python src/train.py --config configs/kan_fkf_full_features.yaml
```

## 📊 Experimental Framework

### Phase 1: Foundational Baselines (Weeks 1-4)
- **MLP Baseline**: Simple feed-forward network with full features
- **Transformer Baseline**: State-of-the-art attention-based model
- **KAN-FKF Baseline**: Primary hypothesis test with FKF architecture

### Phase 2: Feature Ablation & Interpretability (Weeks 5-8)
- **Feature Ablation**: Phonemes-only vs full Booij features
- **Interpretability**: B-spline visualization and attention analysis
- **Progressive Addition**: Stress_level → stress_pos → full_features

### Phase 3: Advanced KAN Architectures (Weeks 9-12)
- **KARN**: Temporal KAN for duration prediction
- **ConvKAN**: 2D KAN for spectrogram generation
- **Hybrid Models**: Integration of specialized KAN variants

### Phase 4: Final Evaluation (Weeks 13-16)
- **Statistical Validation**: Multi-run experiments (5+ runs per model)
- **Subjective Evaluation**: Native Dutch listener MOS studies
- **Comprehensive Analysis**: Thesis-ready results and documentation

## 🔬 Key Features

### Linguistic Feature Integration
- **Booij Stress Marking**: Complete implementation of Dutch phonological rules
- **Hierarchical Data Structure**: Sentence → Word → Phoneme organization
- **Empirical Validation**: TextGrid-based phoneme-syllable alignment
- **Stress Features**: stress_level, primary_stress_pos, secondary_stress_pos

### KAN Architecture Variants
- **FKF (Feed-Forward, KAN, Feed-Forward)**: Primary architecture
- **KARN (Kolmogorov-Arnold Recurrent Network)**: Temporal modeling
- **ConvKAN (Convolutional KAN)**: 2D spatial modeling
- **MultiScaleKAN**: Multi-resolution feature extraction

### Evaluation Metrics
- **Objective**: Duration RMSE, F0 RMSE, Energy RMSE, Mel-Cepstral Distortion
- **Subjective**: Mean Opinion Score (MOS), Intelligibility, A/B Preference
- **Computational**: Training time, inference speed, memory usage

## 📈 Results Tracking

### TensorBoard Integration
```bash
# Monitor training progress
tensorboard --logdir experiments/

# View specific experiment
tensorboard --logdir experiments/20241224_1430_kan_fkf_full_features/logs/
```

### Experiment Comparison
```bash
# Compare multiple experiments
python analysis/compare_pipelines.py \
    --experiments experiments/20241224_1430_baseline_mlp \
                   experiments/20241224_1430_baseline_transformer \
                   experiments/20241224_1430_kan_fkf_full_features
```

## 🔍 Interpretability Analysis

### B-Spline Visualization
```bash
# Visualize learned KAN functions
python analysis/visualize.py \
    --checkpoint experiments/20241224_1430_kan_fkf_full_features/checkpoints/best.pth \
    --feature stress_level \
    --output thesis/figures/bspline_stress.png
```

### Attention Analysis
```bash
# Analyze input relevance
python analysis/visualize.py \
    --checkpoint experiments/20241224_1430_kan_fkf_full_features/checkpoints/best.pth \
    --mode attention \
    --sentence "De Nederlandse taal heeft complexe prosodie." \
    --output thesis/figures/attention_analysis.png
```

## 📚 Academic Documentation

### Research Logbooks
- `thesis/logbooks/STRESSMARKING_LOGBOOK.md`: Complete Booij implementation documentation
- `thesis/logbooks/KAN_CAIN_ANALYSIS_AND_NEXT_STEPS.txt`: Implementation analysis
- `thesis/logbooks/EXPERIMENTAL_FRAMEWORK_LOG.md`: Experimental design documentation

### Thesis Structure
- **Chapter 1**: Introduction to Dutch prosody and English-centric bias
- **Chapter 2**: Data preparation and Booij-validated feature engineering
- **Chapter 3**: Phase 1 results - baseline comparisons
- **Chapter 4**: Phase 2 results - feature ablation and interpretability
- **Chapter 5**: Phase 3 results - advanced KAN architectures
- **Chapter 6**: Final evaluation and conclusions

## 🤝 Contributing

This is a research project for academic thesis work. For questions or collaboration:

1. **Academic Questions**: Contact the research supervisor
2. **Technical Issues**: Open an issue with detailed error information
3. **Reproducibility**: Ensure all experiments are documented with exact configurations

## 📄 License

This project is part of academic research. Please cite appropriately if using this work:

```bibtex
@thesis{mengari2024kan,
  title={Kolmogorov-Arnold Networks for Dutch Text-to-Speech Synthesis with Booij-Validated Linguistic Features},
  author={Mengari, S.},
  year={2024},
  school={University Name}
}
```

## 🔗 Related Work

- **Booij, G. (1995)**: "The Phonology of Dutch" - Linguistic foundation
- **Kolmogorov-Arnold Networks**: Novel neural architecture for function approximation
- **Dutch TTS Research**: Addressing English-centric bias in speech synthesis
- **Interpretable AI**: Making neural networks transparent and explainable

---

**Research Status**: Preprocessing Complete, Neural Training Phase Pending  
**Last Updated**: December 2024  
**Researcher**: S. Mengari  
**Supervisor**: [Supervisor Name] 