# Noise Regime Determines Model Choice

Reproducibility repository for the revised manuscript:

**Noise Regime Determines Model Choice: A Frequency-Selective Robustness Benchmark for Lithium-Ion Battery Prognostics**

Authors: **Alper Bingöl, Mücahit Soylu**

This repository contains the code, frozen protocols, audit artifacts, curated numerical results, and figure sources used in the revised study.

## Main conclusion

The revised benchmark does **not** support a universal architecture winner.

Model behavior depends on the spectral character of sensor error, preprocessing, prediction horizon, and target-domain calibration. In particular, the causal graph model can be advantageous in some conditions, while causal filtering or an MLP can match or outperform it in others.

The term **causal GCN** is used in the manuscript as shorthand for the graph-structured causal temporal model implemented in this study. It should not be interpreted as a conventional symmetric spectral GCN.

## Primary benchmark cohort

The primary cycle-level SOH benchmark pools complete cells from:

| Dataset | Cells | Feature-valid observations |
|---|---:|---:|
| HUST | 77 | 146,122 |
| CALCE | 13 | 14,298 |
| MATR | 169 | 140,133 |
| **Total** | **259** | **300,553** |

RWTH is **not pooled into the primary benchmark** because its target-alignment contract differs from the HUST-CALCE-MATR cohort. RWTH is retained for dataset and alignment auditing where appropriate.

All primary train/validation/test partitions are complete-cell splits.

## Leakage-safe input contract

The primary models use exactly 20 target-independent descriptors:

- 10 voltage descriptors
- 10 current descriptors

For each signal:

`min, q10, q25, median, q75, q90, max, mean, std, phase-slope`

The primary input does **not** include:

- discharge or charge capacity
- SOH
- RUL
- EOL
- cycle number
- observation index
- total cell lifetime
- dataset identity
- future observations

Temperature is excluded from the primary feature contract because a compatible observation contract was not available across the pooled datasets.

dQ/dV features are evaluated separately as a leakage-control experiment rather than being used in the primary benchmark.

## Feature-matched model comparison

The primary MLP and causal graph model both use:

`20 -> 64 -> 64 -> 1`

Each model contains exactly **5,569 trainable parameters**.

The causal graph model applies parameter-free causal mean aggregation before the hidden transforms. Edges connect same-cell predecessors only. There are no future, reverse, or cross-cell edges.

With two aggregation stages, the maximum temporal receptive field is:

`{t, t-1, t-2}`

Training uses:

- AdamW
- learning rate `1e-3`
- weight decay `1e-5`
- no dropout
- no batch normalization
- train-only feature scaling
- hierarchical dataset/cell/observation-balanced MSE
- validation-only early stopping
- patience `30`
- nonbinding maximum of `5000` epochs

The primary experiment uses:

- 5 complete-cell split seeds
- 3 initialization seeds per split
- 15 paired runs
- 30 primary model fits

Initialization seeds are averaged within each split before higher-level paired comparisons.

## Clean SOH benchmark

Balanced cross-dataset results:

| Method | RMSE | MAE |
|---|---:|---:|
| Raw MLP | 0.03798 ± 0.00394 | 0.02642 ± 0.00284 |
| MA3 + MLP | 0.03778 ± 0.00414 | 0.02628 ± 0.00219 |
| LP3 + MLP | 0.03733 ± 0.00430 | 0.02599 ± 0.00245 |
| SG5 + MLP | 0.03862 ± 0.00424 | 0.02710 ± 0.00294 |
| Kalman + MLP | 0.03720 ± 0.00422 | 0.02591 ± 0.00256 |
| Causal GCN | 0.03759 ± 0.00526 | 0.02621 ± 0.00316 |
| Causal TCN | 0.04205 ± 0.00809 | 0.02906 ± 0.00505 |

The five complete-cell splits are treated as repeated-split comparison units, not as five independent datasets.

Exact two-sided sign-flip tests are performed on split-level paired differences. Holm correction is applied separately to the four clean-domain RMSE comparisons and four clean-domain MAE comparisons, and separately to the temporal-control comparison family.

The clean comparison does not establish a statistically supported unique graph advantage.

## Experimental blocks

The revised study contains six main experimental blocks.

### 1. Clean SOH and temporal controls

The feature-matched MLP and causal graph model are compared with generic causal smoothing controls and a causal TCN sequence baseline.

This block tests whether apparent graph-model behavior can instead be explained by generic temporal smoothing.

### 2. Physical-unit sensor robustness

Voltage/current corruption is applied to held-out raw traces before feature recomputation.

Noise families include:

- Gaussian sample noise
- cell-level offset
- causal drift with bounded caps
- dropout with causal filling

Models and train-only scalers remain frozen during these evaluations.

A zero-corruption raw roundtrip is separately audited with a numerical tolerance of `1e-8`.

### 3. Frequency-mechanism diagnostics

Controlled perturbations are applied in the train-only standardized feature space.

These experiments are mechanism diagnostics and are **not** presented as replacements for the physical-unit sensor benchmark.

For the causal mean operator,

`H1(omega) = 0.5 * (1 + exp(-j*omega))`

and

`|H1(omega)| = |cos(omega/2)|`.

Repeated aggregation therefore produces frequency-selective attenuation.

Graph-Fourier analyses use the path Laplacian as a diagnostic tool only; the trained causal operator itself remains directed.

### 4. dQ/dV leakage control

The study separates:

- `PRIMARY20`
- retrospective `FULL` leakage-positive control
- `PREFIX25`
- `PREFIX50`

The prefix conditions use fixed training-derived causal cutoffs so future-cycle information is not available to the model.

### 5. Prospective RUL

Prospective RUL uses an 80%-SOH endpoint.

Exact prospective RUL labels are available for:

- 13 CALCE cells
- 88 MATR cells

HUST is threshold-censored in this experiment and therefore is not used as an exact prospective-RUL target cohort.

### 6. Cross-dataset LODO and few-shot calibration

Leave-one-dataset-out evaluation uses source-only training and source-only feature scaling.

Target-domain calibration follows:

`ZERO -> K1 -> K3 -> K5`

Support-cell selection is deterministic and nested.

Few-shot adaptation:

- starts independently from the zero-shot state
- retains the source-only scaler
- uses learning rate `1e-4`
- uses weight decay `1e-5`
- runs exactly 100 full-support epochs
- uses no target validation early stopping
- does not refit the scaler on target data

Across the audited split-level model/fold/adaptation comparisons, few-shot calibration reduces RMSE relative to zero-shot, while residual cross-domain mismatch remains.

This result does not imply a universal model winner.

## Repository structure

```text
battery-gnn/
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── scripts/
│   └── revision_2026/
├── results/
│   └── revision_2026/
└── figures/
    └── revision_2026/
```

`results/revision_2026/` is a curated public evidence package rather than a dump of every intermediate artifact generated during revision.

The release retains the artifacts needed to audit the final reported results while excluding large regenerated intermediates, model states, raw datasets, and progress/resume-only files.

## Figures

The final figure directory contains publication figures in PDF, PNG, and SVG form together with figure manifests and source-value tables where applicable.

The three final figure groups are:

1. benchmark workflow
2. frequency-selective response
3. LODO/few-shot transfer

## Data availability

Raw third-party battery datasets are **not redistributed in this repository**.

Users should obtain HUST, CALCE, MATR, and RWTH data from their original providers and comply with the corresponding dataset licenses and terms of use.

The repository instead provides dataset audits, reconstruction logic, frozen protocols, hashes, and curated derived results needed to inspect the analysis.

The 185 regenerated MATR per-cell intermediate CSV files are intentionally excluded from the public package. Their reconstruction contract and manifest are retained.

## Reproducibility and provenance

The revision workflow was developed as a sequence of frozen protocols followed by execution and independent audit steps.

Many protocol and execution manifests record SHA-256 hashes of the exact scripts and source artifacts used at the time of execution.

For that reason, the historical revision scripts are retained as **frozen provenance code** and are not rewritten after the fact merely to make their original workspace layout prettier.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before attempting to rerun the full workflow.

## Important interpretation note

Files whose names contain words such as `ranking`, `wins`, or descriptive ordering report point-estimate ordering only unless an inferential test is explicitly attached.

They must not be interpreted as establishing a universal statistically supported model hierarchy.

## Software

The final revision pipeline uses Python 3.10 and the scientific packages listed in `requirements.txt`.

The audited execution environment used Python 3.10.12.

## Release status

This branch corresponds to the revised manuscript analysis and public reproducibility packaging.

The original public repository state and the final revision-analysis state are preserved separately in Git history/tags.
