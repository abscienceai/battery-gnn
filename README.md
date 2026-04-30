# Accuracy Is Not Enough: Robustness-First Evaluation of Temporal Graph Learning for Lithium-Ion Battery Degradation

> **Alper Bingöl & Mücahit Soylu** — İnönü University

This repository contains the code, analysis scripts, and results for our study on temporal graph learning for lithium-ion battery degradation modeling. The central argument is that **clean-distribution accuracy and deployment robustness are distinct optimization objectives** — a model that wins on a pristine benchmark may fail under realistic sensor noise.

---

## Overview

Each battery cell is represented as a directed temporal graph where nodes correspond to individual charge/discharge cycles and edges are weighted by the Pearson correlation between adjacent dQ/dV curves. A Graph Convolutional Network (GCN) is trained on this representation to jointly predict State of Health (SOH) and Remaining Useful Life (RUL) at the node level.

The key experimental finding: under high-frequency stochastic test-time noise, MLP error increases substantially while GCN error increases only modestly. Under low-frequency thermal drift, the ranking reverses. This regime-dependent behavior is the paper's central result.

---

## Repository Structure

```
battery-gnn-main/
├── analysis/
│   └── results/              # All experiment output files (.csv, .json)
├── figures/                  # Generated figures (PNG, 300 DPI)
├── notebooks/                # Jupyter notebooks (exploratory analysis)
├── scripts/                  # All training and evaluation scripts
└── folder_dump.txt           # Full file listing
```

### `analysis/results/`

| File | Description |
|------|-------------|
| `baseline_10seeds.csv` / `baseline_10seeds_fixed.csv` | Main baseline comparison (GCN, MLP, Ridge, XGBoost) over 10 seeds |
| `ablation_10seeds.csv` / `ablation_10seeds_wilcoxon.json` | Ablation study results with Wilcoxon signed-rank tests |
| `scenario_a_10seeds.csv` | Noise robustness — training + test noise (Scenario A) |
| `scenario_b_10seeds.csv` | Noise robustness — test noise only (Scenario B) |
| `advanced_noise_10seeds_long.csv` / `advanced_noise_10seeds_summary.csv` | Realistic BMS sensor degradation scenarios (Scenario C) |
| `gat_10seeds.json` | GAT vs. GCN comparison on the physics-informed temporal graph |
| `gat_noise_scenario_b.json` | GAT noise robustness under Scenario B |
| `ensemble_gat_mlp.json` | GAT + MLP ensemble results |
| `lodo_chemistry_normalized.json` | LODO with per-dataset z-score normalization + chemistry features |
| `fewshot_10seeds.csv` / `fewshot_10seeds_fixed.csv` | Few-shot transfer results (MATR → HUST) |
| `lowdata_10seeds.csv` / `lowdata_10seeds_fixed.csv` / `lowdata_10seeds_globalnorm.csv` | Low-data regime experiments |
| `early_prediction.csv` / `early_prediction_10seeds.csv` | Early-cycle prediction (10% and 20% observed cycles) |
| `rul_all_models.csv` | RUL prediction comparison across all model families |
| `rul_scenario_a_10seeds.csv` / `rul_scenario_b_10seeds.csv` | RUL noise robustness (Scenarios A and B) |
| `per_dataset_predictions.csv` | Per-dataset (MATR, CALCE, HUST, RWTH) SOH predictions |
| `dann_results.csv` | Domain Adaptation (DANN) experiment results |
| `noise_robustness.csv` | Single-seed noise robustness baseline |
| `nodelevel_weighted_metrics.json` | Node-level multi-task GCN metrics (SOH + RUL) |
| `p2_amplitude_normalized.csv` | Amplitude-normalized dQ/dV control experiment |
| `test_noise_only.csv` / `test_noise_only_seeds.csv` | Test-noise-only scenario variants |
| `advanced_noise_results.csv` | Single-seed advanced noise experiment |
| `few_shot_results.json` | Few-shot transfer JSON output |
| `ablation_no_cycle_graph.json` / `ablation_no_dqdv.json` / `ablation_no_gnn.json` | Individual ablation variant outputs |

### `figures/`

| File | Description |
|------|-------------|
| `fig2_capacity_degradation_curves.png` | Capacity fade curves for representative cells from each dataset |
| `fig3_graph_construction_paradigm.png` | Physics-informed temporal graph vs. conventional chain graph |
| `fig4_dqdv_rwth_peaks.png` | dQ/dV curve with detected peaks for an NMC (RWTH) cell |
| `fig5_training_validation_loss.png` | GCN training and validation loss curves |
| `fig6_gnnexplainer_node_importance.png` | GNNExplainer cycle-level node importance scores |
| `fig7_gnnexplainer_edge_importance.png` | GNNExplainer edge importance across consecutive cycles |
| `fig8_tsne_domain_visualization.png` | t-SNE projection of GCN node embeddings by dataset |
| `fig9_shap_feature_importance.png` | SHAP summary plot of input feature contributions |
| `fig10_pareto_frontier.png` | Deployment–accuracy trade-off (Pareto frontier) |
| `fig11_noise_robustness_curve.png` | SOH MAE vs. noise level for Scenarios A and B |
| `fig12_cross_dataset_r2_heatmap.png` | Cross-dataset R² heatmap |

### `scripts/`

#### Graph Construction & Preprocessing
| Script | Description |
|--------|-------------|
| `build_cycle_graphs.py` | Builds temporal cycle graphs from raw battery CSV data; computes dQ/dV features and RUL targets; saves train/val/test splits as `.pkl` files |

#### Main Model Training
| Script | Description |
|--------|-------------|
| `train_gcn_no_leakage.py` | Trains the primary GCN model (no target leakage); saves `gcn_no_leakage.pth` |
| `train_baselines_no_leakage.py` | Trains MLP, Ridge, and XGBoost baselines on the same leakage-free splits |
| `train_nodelevel_weighted.py` | Multi-task node-level GCN (SOH + RUL joint training) |
| `train_lstm_no_leakage.py` | Cell-level LSTM baseline |
| `train_lstm_cell_based.py` | Cell-based LSTM with sequence padding |
| `train_cnn_no_leakage.py` | 1D CNN baseline |

#### Baseline & Ablation Studies
| Script | Description |
|--------|-------------|
| `baseline_10seeds.py` | 10-seed baseline comparison (GCN, MLP, Ridge, XGBoost) |
| `ablation_10seeds.py` | 10-seed ablation: full model vs. no-cycle-graph vs. no-dQ/dV vs. no-GNN |
| `Ablasyon10seedWilcoxon.py` | Ablation with Wilcoxon signed-rank significance tests |

#### Noise Robustness Experiments
| Script | Description |
|--------|-------------|
| `scenario_a_10seeds.py` | Scenario A: training + test Gaussian noise, 10 seeds |
| `test_noise_only_10seeds.py` | Scenario B: test-only Gaussian noise, 10 seeds |
| `test_noise_only.py` | Single-seed test-noise-only evaluation |
| `test_noise_only_seeds.py` | Multi-seed test-noise-only variant |
| `scenario_a_3seeds.py` | Scenario A with 3 seeds (exploratory) |
| `gat_noise_scenario_b.py` | GAT under Scenario B noise |
| `frequency_noise_experiment.py` | Frequency-dependent noise: white vs. low-frequency correlated noise |
| `test_advanced_noise.py` | Realistic BMS noise (thermal drift, calibration offset, sensor dropout, compound) |
| `test_advanced_noise_10seeds.py` | Same as above over 10 seeds with Wilcoxon tests |
| `noise_robustness.py` | Single-run noise robustness sweep |

#### GAT & Ensemble
| Script | Description |
|--------|-------------|
| `gat_experiment.py` | GAT (3-layer, 4-head, edge dropout) over 10 seeds |
| `ensemble_gat_mlp.py` | Weighted ensemble of GAT and MLP predictions |

#### RUL Estimation
| Script | Description |
|--------|-------------|
| `rul_all_models.py` | RUL prediction comparison: MLP, GCN, Ridge, XGBoost |
| `rul_noise_scenario_a_10seeds.py` | RUL noise robustness — Scenario A |
| `rul_noise_scenario_b_10seeds.py` | RUL noise robustness — Scenario B |

#### Transfer Learning & Domain Adaptation
| Script | Description |
|--------|-------------|
| `few_shot_transfer.py` | Few-shot fine-tuning: MATR (source) → HUST (target) |
| `fewshot_10seeds.py` | Same over 10 seeds with zero-shot and finetune ratios |
| `train_dann_all.py` | Domain Adversarial Neural Network (DANN) across all four datasets |
| `lodo_experiment.py` | Leave-One-Dataset-Out (LODO) evaluation |
| `lodo_chemistry_normalized.py` | LODO with chemistry features + per-dataset z-score normalization |

#### Low-Data & Early Prediction
| Script | Description |
|--------|-------------|
| `lowdata_10seeds.py` | Low-data regime: 1%, 5%, 10% training data over 10 seeds |
| `early_prediction.py` | Early-cycle prediction with 10% and 20% observed cycles |
| `early_prediction_10seeds.py` | Same over 10 seeds |

#### Interpretability & Visualization
| Script | Description |
|--------|-------------|
| `gnn_explainer.py` | GNNExplainer node and edge importance for an RWTH test cell |
| `shap_analysis.py` | SHAP KernelExplainer on 200 test nodes |
| `tsne_visualization.py` | t-SNE projection of GCN node embeddings |
| `compute_per_domain_correct.py` | Per-domain (MATR/CALCE/HUST/RWTH) SOH evaluation metrics |
| `compute_per_domain_metrics.py` | Per-domain metrics with saved model checkpoint |
| `evaluate_per_dataset.py` | Per-dataset SOH MAE and R² |

#### Figure Generation
| Script | Description |
|--------|-------------|
| `plot_all_figures.py` | Generates all main paper figures |
| `plot_capacity_fade.py` | Capacity fade curves by dataset |
| `plot_loss_curves.py` | Training and validation loss curves |
| `plot_architecture.py` | Multi-task GNN architecture diagram |
| `create_figure3.py` | Graph construction paradigm comparison figure |

#### Feature Engineering
| Script | Description |
|--------|-------------|
| `p2_amplitude_normalized.py` | Amplitude-normalized dQ/dV control experiment |
| `edge_weight_cosine_euclidean.py` | Edge weight comparison: Pearson vs. cosine vs. Euclidean vs. fixed |
| `edge_weight_pearson_vs_dtw.py` | Pearson vs. DTW edge weight comparison |
| `train_dqdv_rwth_only.py` | Ablation: dQ/dV features restricted to RWTH (NMC) only |

---

## Datasets

Four open-access datasets from the [BatteryML](https://github.com/microsoft/BatteryML) platform are used:

| Dataset | Cells | Cycles | Chemistry | Nominal Capacity |
|---------|-------|--------|-----------|-----------------|
| [MATR](https://doi.org/10.1038/s41560-019-0356-8) | 180 | 154,107 | LFP/graphite | 1.10 Ah |
| [CALCE](https://calce.umd.edu/battery-data) | 13 | 14,298 | LCO/graphite | 1.10 Ah |
| [HUST](https://doi.org/10.17632/nsc7hnsg4s.2) | 77 | 64,441 | LFP/graphite | 1.10 Ah |
| [RWTH](https://doi.org/10.1016/j.est.2015.05.003) | 48 | 104,959 | NMC/carbon | 1.11 Ah |

> **Note:** HUST is used as a low-variance diagnostic documentation case. Its discharge capacity targets are clustered in the 0–0.05 Ah range, making standard metrics such as MAE and R² unreliable for primary comparison. MATR, CALCE, and RWTH form the main quantitative benchmark.

Place the raw data CSV at:
```
analysis/battery_data_all_complete.csv
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build cycle graphs
```bash
python scripts/build_cycle_graphs.py
```
This reads `analysis/battery_data_all_complete.csv`, computes dQ/dV features, and saves `train_graphs_no_target.pkl`, `val_graphs_no_target.pkl`, and `test_graphs_no_target.pkl` to `analysis/cycle_graphs/`.

### 3. Train the primary GCN model
```bash
python scripts/train_gcn_no_leakage.py
```

### 4. Run the main baseline comparison (10 seeds)
```bash
python scripts/baseline_10seeds.py
```

### 5. Run noise robustness experiments
```bash
# Scenario A: training + test noise
python scripts/scenario_a_10seeds.py

# Scenario B: test noise only
python scripts/test_noise_only_10seeds.py

# Scenario C: realistic BMS sensor degradation
python scripts/test_advanced_noise_10seeds.py
```

### 6. Run ablation studies
```bash
python scripts/ablation_10seeds.py
```

### 7. Generate figures
```bash
python scripts/plot_all_figures.py
```

---

## Model Architecture

The primary model is a 3-layer GCN with two regression heads:

- **Input:** Node features — temperature, current, nominal capacity, voltage limits, and chemistry-aware dQ/dV descriptors (10 features total). The target variable (discharge capacity) is **excluded** from node features to prevent leakage.
- **GCN layers:** 3 × `GCNConv(hidden_dim=128)` with ReLU activations
- **Heads:** SOH head (128→1) and RUL head (128→1)
- **Loss:** `L = MSE(SOH) + 0.01 × MSE(RUL)`
- **Optimizer:** Adam, lr=1e-3, batch=32, 50 epochs
- **Split:** 70% train / 15% validation / 15% test (cell-level)

Graph Attention Network (GAT) variants are also provided (`gat_experiment.py`, `ensemble_gat_mlp.py`) and use the same physics-informed temporal graph with identical hyperparameters.

---

## Key Results

| Model | SOH MAE (Ah) | SOH R² |
|-------|-------------|--------|
| MLP | 0.0618 ± 0.0007 | 0.9172 ± 0.0012 |
| XGBoost | 0.0624 | 0.9145 |
| GCN | 0.0632 ± 0.0007 | 0.9097 ± 0.0010 |
| Ridge | 0.0687 | 0.8894 |

Under high-frequency test-time noise (Scenario B, σ=0.10): MLP MAE increases by **24.9%** while GCN MAE increases by only **5.8%**. Under low-frequency thermal drift, the ranking reverses in favor of MLP.

---

## Reproducibility

All stochastic experiments are repeated over 10 fixed seeds: `[5782, 327, 984, 2430, 3382, 8039, 6182, 5398, 6699, 7656]`. Results are reported as mean ± standard deviation. Statistical significance is assessed with Wilcoxon signed-rank tests.

Trained model checkpoints:
- `analysis/results/gcn_no_leakage.pth`
- `analysis/results/mlp_no_leakage.pth`
- `analysis/results/nodelevel_best.pth`

---

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{bingol2025accuracy,
  title   = {Accuracy Is Not Enough: Robustness-First Evaluation of Temporal Graph Learning for Lithium-Ion Battery Degradation},
  author  = {Bing{\"o}l, Alper and Soylu, M{\"u}cahit},
  journal = {Preprint},
  year    = {2025}
}
```

---

## License

This project is released for academic and research use. Please refer to the individual dataset licenses (MATR, CALCE, HUST, RWTH) for data usage terms.
