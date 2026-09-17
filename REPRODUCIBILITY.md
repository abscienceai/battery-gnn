# Reproducibility and provenance

## Purpose

This repository distinguishes between:

1. **frozen scientific provenance**, and
2. **public release packaging**.

The distinction is intentional.

During revision, protocols were frozen before corresponding test analyses, model executions were audited, and many manifests stored SHA-256 hashes of the exact scripts and artifacts used.

Changing those historical scripts after the analysis would make the published code look cleaner while breaking the recorded provenance chain.

Therefore, the scientific scripts under:

```text
scripts/revision_2026/
```

should be treated as frozen executed-analysis code.

## Original revision workspace

The executed revision workflow used a workspace of the form:

```text
battery_revision_2026/
├── data/
├── results/
├── export/
└── repo/
    └── battery-gnn/
```

Several frozen scripts encode assumptions about this original research workspace.

These path assumptions are not evidence of data leakage and are not inputs to the trained models. They describe where source data, intermediate artifacts, results, and the repository were located during execution.

A small number of early audit/provenance scripts may contain an original machine-specific path. These are retained because their hashes are referenced by audit artifacts.

## Public result tree

For the GitHub release, final results are curated into:

```text
results/revision_2026/
```

The public result tree is intentionally much smaller than the complete private evidence bundle.

The curated release excludes, among other items:

- 185 regenerated MATR per-cell intermediate CSV files
- raw battery datasets
- trained model-state files
- heavy prediction dumps
- per-cell physical-noise provenance files containing original raw-source paths
- progress/resume-only artifacts
- superseded validation-calibration runs
- superseded clean-benchmark versions

## Path sanitization

Public copies of selected JSON/CSV provenance artifacts were generated from the verified evidence bundle.

Absolute revision-workspace prefixes in the public copies were replaced with:

```text
${WORKSPACE}
```

The source artifact SHA-256 and the sanitized public artifact SHA-256 are both recorded in:

```text
results/revision_2026/RELEASE_SOURCE_INDEX.tsv
```

This allows the public packaging transformation itself to be audited without pretending that a sanitized copy is byte-identical to the original evidence artifact.

## Evidence integrity

Before public packaging, the complete evidence bundle index was independently checked.

Verified bundle status:

```text
checked:       1506
missing:       0
size mismatch: 0
hash mismatch: 0
status:        PASS
```

This establishes byte-level agreement with the recorded evidence index. It does not by itself substitute for scientific interpretation or methodological review.

## Split and preprocessing discipline

The primary SOH benchmark uses complete-cell train/validation/test splits.

Feature preprocessing is fit on training data only and then frozen for validation/test use.

The five split seeds are repeated-split comparison units.

Three initialization seeds are averaged within each split before paired higher-level comparison.

## Statistical comparison

The main paired sensitivity analyses use exact two-sided sign-flip tests on the five split-level differences.

Holm correction is applied within the pre-specified metric/comparison families.

The repeated splits are not claimed to represent five statistically independent datasets.

## Physical-noise versus frequency-mechanism experiments

These are deliberately separate experimental contracts.

The physical-noise benchmark corrupts raw held-out voltage/current traces in physical units and then recomputes descriptors.

The frequency-mechanism experiments perturb train-standardized descriptor space to investigate frequency-selective behavior.

Feature-space spectral experiments are therefore mechanism diagnostics, not substitutes for physical sensor realism.

## dQ/dV leakage control

dQ/dV is not a primary input feature.

The retrospective FULL condition is intentionally leakage-positive and is used as a control.

PREFIX25 and PREFIX50 use training-derived fixed causal cutoffs to prevent access to future trajectory information.

## Prospective RUL

The prospective-RUL experiment is restricted to cells for which the 80%-SOH endpoint can be established under the frozen contract.

HUST is threshold-censored and is not treated as an exact RUL target cohort.

## LODO and few-shot transfer

The LODO experiment retains source-only scalers during target-domain adaptation.

K1, K3, and K5 support sets are nested and selected deterministically.

Target validation is not used during few-shot adaptation.

## Raw data

This repository does not redistribute raw third-party datasets.

To reproduce the complete pipeline from raw observations, obtain each dataset from its original provider and comply with its license/terms.

Because dataset archives and directory layouts can change independently of this repository, users should inspect the corresponding dataset-audit scripts before reconstructing the original workspace.

## Recommended use of this repository

For result inspection:

1. read the frozen protocol
2. read the corresponding execution/audit manifest
3. inspect the compact analysis tables
4. use the release source index to distinguish original evidence hashes from sanitized public-copy hashes

For a full raw-data rerun, reproduce the required workspace and dataset layout in a separate clone/copy rather than editing the frozen provenance scripts in place.
