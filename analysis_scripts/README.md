# Interpretability Analysis Scripts

## **Purpose**
This directory contains scripts for analyzing trained KAN models and providing evidence for the interpretability claims in the thesis.

## **Scripts to be Developed** (After Training)

### **1. B-Spline Visualization** (`visualize_bsplines.py`)
- **Purpose**: Analyze what rules the KAN learned from Booij features
- **Input**: Trained `kan_fkf_full_features` model checkpoints
- **Output**: Visualizations of learned B-spline functions
- **Academic Value**: Direct evidence of KAN interpretability

### **2. Input Relevance Analysis** (`input_relevance.py`)
- **Purpose**: Compare full-featured vs. phonemes-only models
- **Input**: Both trained model checkpoints and test data
- **Output**: Feature attribution and importance scores
- **Academic Value**: Evidence for why Booij features matter

### **3. Feature Attribution** (`feature_attribution.py`)
- **Purpose**: Understand which linguistic features contribute most to prosody
- **Input**: Trained models and ablation results
- **Output**: Feature importance rankings and visualizations
- **Academic Value**: Validation of H2 hypothesis

### **4. Model Comparison** (`model_comparison.py`)
- **Purpose**: Systematic comparison of MLP, Transformer, KAN baselines
- **Input**: All Phase 1 experiment results
- **Output**: Performance metrics and statistical analysis
- **Academic Value**: Baseline establishment for thesis

## **Dependencies**
- Trained model checkpoints from cloud experiments
- Experiment logs and metrics
- Test datasets for evaluation
- Visualization libraries (matplotlib, seaborn, plotly)

## **Execution Order**
1. Wait for cloud training completion
2. Download model checkpoints and logs
3. Develop and run analysis scripts
4. Generate visualizations and insights
5. Document findings in thesis

## **Academic Rigor Notes**
- All analyses will be reproducible with provided code
- Statistical significance will be tested where appropriate
- Visualizations will be publication-ready
- Results will directly address thesis hypotheses 