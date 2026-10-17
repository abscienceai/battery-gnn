# Operational battery data and field-informed sensitivity analysis

Reproducibility materials for the field-data characterization
and exploratory perturbation experiment reported in the
manuscript *Noise Regime Determines Model Choice*.

## Operational datasets

1. Energy Storage Power Station dataset:
   https://doi.org/10.5281/zenodo.18471156
2. Xie operational battery dataset:
   https://doi.org/10.5281/zenodo.18328701
3. Zhang et al. EV fault-detection dataset, Brand 3 subset:
   https://doi.org/10.6084/m9.figshare.23659323

Original third-party battery time series are not redistributed.

## Analysis workflow

- 04: Characterize short-timescale Li/Xie voltage increments.
- 05: Verify raw-feature reconstruction and pilot predictions.
- 06: Extend the pilot across eligible held-out HUST cells.
- 07: Characterize processed Zhang charging records.
- 08: Perform the Gaussian comparator.
- 09: Reproduce 10,000-replicate cell-cluster bootstrap.

The experimental scripts 04-08 use the original project
workspace paths, raw source datasets, frozen model checkpoints,
split assignment, train-only scaler, and canonical feature
pipeline. They are not standalone models and do not operate
on the public operational datasets as direct SOH test targets.

The bootstrap script works from the saved pilot CSV outputs.
The release includes these pilot outputs and aggregate
statistical summaries.

To run the bootstrap directly within the original workspace:

    export R2=/path/to/EST_R2_field_validation_2026-10
    python scripts/09_cell_cluster_bootstrap.py

Store the released results CSV files in $R2/reports.
The bootstrap uses NumPy and pandas.

## Experimental contract

- 11 HUST held-out cells, 40 eligible observations per cell.
- One fixed split and model initialization.
- Frozen matched MLP and causal GCN models.
- Three synthetic perturbation seeds per source profile.
- Voltage-only hypothetical jitter before feature recomputation.
- Gaussian comparator and 10,000 cell-cluster bootstrap samples.
- Source measurements DO NOT provide verified sensor-error labels.

The Li/Xie increment residuals may contain genuine cell
heterogeneity, quantization and operating-condition effects.
Their use as level jitter is a synthetic sensitivity analysis,
not empirical sensor-fault validation.

## Scope and data provenance

Source files are obtained from their original publishers.
No complete raw operational EV or power-station time series,
battery measurement archives, or personal information are
distributed here.

The original model and preprocessing code are in the parent
repository. Reconstructing upstream analyses requires the
source datasets and project-specific frozen artifacts.

## Published Frozen Evaluation Artifacts

The `results/revision_2026/field_operational/artifacts/`
directory provides the frozen model state dictionaries
for the matched MLP and causal GCN, train-only scaler
parameters, and original cell-split assignment information.

The original laboratory measurements and complete
harmonized canonical training table are not redistributed.

Reconstructing the raw-signal evaluation requires the
upstream laboratory datasets, the original feature-generation
pipeline, and the corresponding canonical data artifacts.

The experimental scripts retain the research workspace
structure and are not a standalone one-command
reproduction package.

The published per-cell prediction results, Gaussian-control
comparisons, and cell-cluster bootstrap implementation
allow independent verification of the reported statistical
comparisons without retraining the models.

The Zhang segment and vehicle profiles describe processed
operational electric-vehicle charging records. Vehicle
fault labels are not independently verified sensor-fault labels.

## Reproducibility Limitations

The field-derived voltage increments are not ground-truth
sensor errors.

The exploratory sensitivity analysis is limited to
11 held-out HUST cells, one frozen model pair,
and three synthetic perturbation repetitions per profile.

The results do not establish robustness against
independently verified real-world BMS sensor failures.

## Relevance to Electric Vehicles and Intelligent Energy Systems

Reliable battery state-of-health (SOH) estimation is a
fundamental component of battery monitoring, electric
mobility, and electrochemical energy-storage applications.

As electric vehicles and distributed charging infrastructure
become increasingly interconnected, the reliability of
battery-health predictions under imperfect measurements
and changing operating conditions becomes especially
important.

The robustness-first methodology investigated in this
repository provides insights into how sensor perturbations,
temporal measurement characteristics, and domain shifts
can influence battery prognostics.

These considerations have broader relevance to intelligent
charging systems, battery-aware energy management,
and dependable operation of electric mobility infrastructure.

### Potential Applications and Future Research

Several application directions arise naturally from
the challenges investigated in this study:

- **Electric vehicle battery monitoring:** Developing
  dependable SOH estimation and degradation-awareness
  methods under uncertain measurement conditions.

- **Intelligent charging:** Exploring how reliable battery
  health information could inform charging optimization,
  scheduling, and operational decision support.

- **Distributed energy systems:** Investigating
  battery-condition awareness in charging networks,
  stationary storage, and interconnected energy resources.

- **Resilient charging infrastructure:** Studying how
  battery reliability and charging-resource availability
  may contribute to continuity of essential mobility
  services during infrastructure disruptions.

- **Emergency mobility and energy resilience:**
  Exploring data-driven approaches for assessing
  charging readiness, resource availability, and
  potential operational constraints during emergencies.

- **Digital twins and predictive analytics:** Investigating
  future integration of degradation-aware battery models
  with simulations of electric-vehicle charging systems
  and infrastructure operating scenarios.

### Scientific Scope

The present repository reports battery prognostics,
causal temporal modeling, sensor-robustness experiments,
and exploratory operational battery-data analyses.

It does not currently implement or validate an
infrastructure-scale charging coordinator,
emergency mobility management platform, or
charging-network digital twin.

The applications outlined above identify potential
research directions rather than experimentally
demonstrated outcomes of the present study.
