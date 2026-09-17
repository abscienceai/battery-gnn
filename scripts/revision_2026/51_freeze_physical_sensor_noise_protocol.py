from pathlib import Path
import hashlib
import json
import os
import subprocess


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo" / "battery-gnn"

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "physical_sensor_noise"
    / "final_physical_v1"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PROTOCOL_OUT = ROOT / "protocol.json"


SOURCES = {
    "cycle_feature_kernel":
        REPO
        / "scripts"
        / "revision_2026"
        / "cycle_feature_kernel.py",

    "raw_cycle_generator":
        REPO
        / "scripts"
        / "revision_2026"
        / "14_raw_cycle_feature_generator.py",

    "canonical_builder":
        REPO
        / "scripts"
        / "revision_2026"
        / "20_canonical_task_alignment_builder.py",

    "benchmark_model":
        REPO
        / "scripts"
        / "revision_2026"
        / "24_feature_matched_soh_benchmark.py",

    "canonical_table":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "canonical_tasks"
        / "canonical_cycle_supervised_table.csv",

    "split_assignments":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "splits_and_scalers"
        / "cell_split_assignments.csv",

    "train_only_scalers":
        BASE
        / "data"
        / "processed"
        / "revision_2026"
        / "splits_and_scalers"
        / "train_only_scaler_parameters.csv",

    "clean_protocol":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "full_clean_benchmark_protocol_v3.json",

    "clean_execution_manifest":
        BASE
        / "results"
        / "revision_2026"
        / "feature_matched_soh"
        / "final_v3"
        / "final_v3_execution_manifest.json",
}


EXPECTED_IMMUTABLE_HASHES = {
    "canonical_table":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "split_assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "train_only_scalers":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "clean_protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "clean_execution_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",
}


PRIMARY_FEATURES = [
    "v_min",
    "v_q10",
    "v_q25",
    "v_median",
    "v_q75",
    "v_q90",
    "v_max",
    "v_mean",
    "v_std",
    "v_phase_slope",

    "i_min",
    "i_q10",
    "i_q25",
    "i_median",
    "i_q75",
    "i_q90",
    "i_max",
    "i_mean",
    "i_std",
    "i_phase_slope",
]


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def atomic_json(obj, path):
    tmp = Path(
        str(path)
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


# ============================================================================
# Source existence + fixed historical artifact guards
# ============================================================================

for name, path in SOURCES.items():
    assert path.is_file(), (
        name,
        path,
    )


for name, expected in EXPECTED_IMMUTABLE_HASHES.items():

    observed = sha256_file(
        SOURCES[name]
    )

    assert observed == expected, (
        name,
        observed,
        expected,
    )


# Source scripts that define the raw -> V/A -> feature contract
# must be tracked and unmodified at protocol freeze.
tracked_sources = [
    "scripts/revision_2026/cycle_feature_kernel.py",
    "scripts/revision_2026/14_raw_cycle_feature_generator.py",
    "scripts/revision_2026/20_canonical_task_alignment_builder.py",
    "scripts/revision_2026/24_feature_matched_soh_benchmark.py",
]


for rel in tracked_sources:

    r = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            rel,
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    assert r.returncode == 0, rel


    r = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            rel,
        ],
        cwd=REPO,
    )

    assert r.returncode == 0, (
        "modified tracked source",
        rel,
    )


# ============================================================================
# No physical-noise evaluation may predate this protocol
# ============================================================================

for pattern in [
    "test_metrics*.csv",
    "evaluation*.json",
    "corrupted_features*.csv",
    "corrupted_features*.parquet",
    "analysis_manifest.json",
    "execution_manifest.json",
]:

    found = list(
        ROOT.rglob(
            pattern
        )
    )

    assert len(found) == 0, (
        pattern,
        found[:5],
    )


# ============================================================================
# Frozen corruption definitions
# ============================================================================

levels = {
    "low": {
        "severity_axis":
            1.0 / 3.0,

        "gaussian_voltage_sigma_V":
            0.002,

        "gaussian_current_sigma_A":
            0.005,

        "calibration_voltage_sigma_V":
            0.002,

        "calibration_current_sigma_A":
            0.005,

        "drift_voltage_per_100_cycles_V":
            0.001,

        "drift_voltage_cap_V":
            0.005,

        "drift_current_per_100_cycles_A":
            0.002,

        "drift_current_cap_A":
            0.010,

        "dropout_probability":
            0.005,
    },

    "medium": {
        "severity_axis":
            2.0 / 3.0,

        "gaussian_voltage_sigma_V":
            0.005,

        "gaussian_current_sigma_A":
            0.010,

        "calibration_voltage_sigma_V":
            0.005,

        "calibration_current_sigma_A":
            0.010,

        "drift_voltage_per_100_cycles_V":
            0.002,

        "drift_voltage_cap_V":
            0.010,

        "drift_current_per_100_cycles_A":
            0.005,

        "drift_current_cap_A":
            0.025,

        "dropout_probability":
            0.010,
    },

    "high": {
        "severity_axis":
            1.0,

        "gaussian_voltage_sigma_V":
            0.010,

        "gaussian_current_sigma_A":
            0.025,

        "calibration_voltage_sigma_V":
            0.010,

        "calibration_current_sigma_A":
            0.025,

        "drift_voltage_per_100_cycles_V":
            0.005,

        "drift_voltage_cap_V":
            0.020,

        "drift_current_per_100_cycles_A":
            0.010,

        "drift_current_cap_A":
            0.050,

        "dropout_probability":
            0.030,
    },
}


protocol = {
    "contract":
        "physical_sensor_noise_soh_v1",

    "status":
        "FINAL_FROZEN_BEFORE_ANY_PHYSICAL_SENSOR_CORRUPTION_EVALUATION",

    "purpose":
        (
            "Replace normalized feature-space perturbation with "
            "raw-input sensor corruption in physical units, followed "
            "by complete feature recomputation through the frozen "
            "cycle feature kernel."
        ),

    "primary_task":
        "cycle-level SOH regression",

    "datasets":
        [
            "HUST",
            "CALCE",
            "MATR",
        ],

    "models":
        [
            "FeatureMatchedMLP",
            "FeatureMatchedCausalGCN",
        ],

    "model_source":
        "frozen final_v3 clean-trained states; no noisy retraining",

    "model_parameter_count_each":
        5569,

    "split_seeds":
        [
            0,
            1,
            2,
            3,
            4,
        ],

    "model_initialization_seeds":
        [
            0,
            1,
            2,
        ],

    "corruption_seeds":
        [
            0,
            1,
            2,
            3,
            4,
        ],

    "primary_feature_count":
        20,

    "primary_features":
        PRIMARY_FEATURES,

    "raw_model_input_channels":
        {
            "voltage":
                {
                    "used":
                        True,

                    "physical_unit":
                        "V",
                },

            "current":
                {
                    "used":
                        True,

                    "physical_unit":
                        "A",
                },

            "within_cycle_time":
                {
                    "used":
                        True,

                    "physical_unit":
                        "s",

                    "corrupted":
                        False,

                    "role":
                        (
                            "chronology and normalized within-cycle "
                            "phase only"
                        ),
                },

            "temperature":
                {
                    "used":
                        False,

                    "corrupted_in_primary_benchmark":
                        False,

                    "reason":
                        (
                            "Temperature is not part of the frozen "
                            "20-feature primary model input. A "
                            "temperature corruption would therefore "
                            "not propagate into this model."
                        ),
                },

            "capacity":
                {
                    "used":
                        False,

                    "corrupted_in_primary_benchmark":
                        False,

                    "reason":
                        (
                            "Capacity/SOH belongs to the supervised "
                            "target definition and must not be "
                            "introduced as a corrupted model input; "
                            "doing so would change target semantics "
                            "rather than test input-sensor robustness."
                        ),
                },
        },

    "unit_conversion_rule":
        (
            "Use the frozen dataset adapters to convert source-native "
            "sensor channels to voltage in V, current in A, and time "
            "in s first. Apply corruption only after this conversion "
            "and before feature extraction."
        ),

    "feature_recomputation_rule":
        (
            "For every corrupted raw cycle, call the same frozen "
            "extract_primary_cycle_features(voltage_V,current_A,time_s) "
            "kernel used by the clean revision pipeline. No direct "
            "perturbation of canonical/model feature tensors is allowed."
        ),

    "old_feature_space_noise_role":
        (
            "Legacy g.x-space perturbation is not used as the primary "
            "physical-sensor robustness evidence."
        ),

    "training_policy":
        {
            "no_retraining":
                True,

            "no_noise_augmentation_training":
                True,

            "frozen_clean_trained_models":
                True,

            "test_only_corruption":
                True,

            "train_only_scaler_reused_without_refit":
                True,

            "validation_or_test_statistics_may_refit_scaler":
                False,
        },

    "clean_roundtrip_barrier":
        {
            "required_before_noisy_test_metrics":
                True,

            "rule":
                (
                    "Reconstruct each eligible test cycle from raw "
                    "source data with zero corruption, run the frozen "
                    "cycle kernel, align to the canonical row, and "
                    "verify the 20 features against the existing clean "
                    "canonical features before any noisy model metric "
                    "is evaluated."
                ),

            "feature_tolerance":
                {
                    "absolute":
                        1e-8,

                    "relative":
                        1e-8,
                },

            "mapping_requirements":
                {
                    "future_cycle_use":
                        False,

                    "cross_cell_mapping":
                        False,

                    "target_used_for_feature_generation":
                        False,
                },

            "on_failure":
                (
                    "Stop before noisy evaluation and audit dataset "
                    "adapter/alignment; do not relax tolerance based "
                    "on noisy test performance."
                ),
        },

    "corruption_application_scope":
        {
            "phase":
                "test only",

            "sample_level_gaussian":
                True,

            "cell_level_calibration_bias":
                True,

            "cycle_progressive_sensor_drift":
                True,

            "sample_dropout":
                True,

            "corruption_occurs_before_feature_recomputation":
                True,
        },

    "severity_levels":
        levels,

    "scenario_families":
        {
            "gaussian":
                {
                    "levels":
                        [
                            "low",
                            "medium",
                            "high",
                        ],

                    "definition":
                        (
                            "At every raw sample independently add "
                            "zero-mean Gaussian noise in V and A using "
                            "the level-specific physical-unit sigma."
                        ),
                },

            "calibration_offset":
                {
                    "levels":
                        [
                            "low",
                            "medium",
                            "high",
                        ],

                    "definition":
                        (
                            "For each dataset/cell/corruption seed draw "
                            "one zero-mean Gaussian voltage bias and one "
                            "current bias using the level-specific sigma; "
                            "hold each bias constant across all samples "
                            "and cycles of that cell."
                        ),
                },

            "sensor_drift":
                {
                    "levels":
                        [
                            "low",
                            "medium",
                            "high",
                        ],

                    "definition":
                        (
                            "For each cell and corruption seed draw the "
                            "drift sign independently for V and I. "
                            "Bias magnitude grows with observed cycle "
                            "index at the specified physical-unit rate "
                            "per 100 cycles and is clipped at the "
                            "level-specific absolute cap. No total "
                            "future lifetime is used."
                        ),
                },

            "sample_dropout":
                {
                    "levels":
                        [
                            "low",
                            "medium",
                            "high",
                        ],

                    "definition":
                        (
                            "Apply Bernoulli sample dropout independently "
                            "within each raw cycle. Protect the first valid "
                            "sample, then causally forward-fill missing "
                            "V/I observations from the most recent retained "
                            "sample. Time coordinates remain unchanged."
                        ),
                },

            "compound_medium":
                {
                    "levels":
                        [
                            "medium",
                        ],

                    "definition":
                        (
                            "Apply medium calibration bias plus medium "
                            "sensor drift plus medium Gaussian sample "
                            "noise, followed by medium sample dropout "
                            "and causal forward-fill."
                        ),

                    "inferential_role":
                        "descriptive stress test only",
                },
        },

    "compound_operation_order":
        [
            "calibration_offset",
            "sensor_drift",
            "gaussian",
            "sample_dropout_and_causal_forward_fill",
        ],

    "randomness_contract":
        {
            "stable_hash_seeded":
                True,

            "seed_key_fields":
                [
                    "corruption_seed",
                    "scenario",
                    "level",
                    "dataset",
                    "cell_id",
                    "cycle_id",
                ],

            "global_rng_state_dependency":
                False,

            "same_corruption_realization_for_mlp_and_gcn":
                True,
        },

    "evaluation_contract":
        {
            "test_targets_unchanged":
                True,

            "test_cell_membership_unchanged":
                True,

            "train_only_scaler_reused":
                True,

            "model_weights_unchanged":
                True,

            "metrics":
                [
                    "RMSE",
                    "MAE",
                    "R2_by_dataset",
                ],

            "balanced_all":
                (
                    "same dataset-balanced and cell-balanced weighting "
                    "as the frozen clean benchmark"
                ),
        },

    "aggregation_contract":
        {
            "within_split":
                (
                    "Average three model initialization seeds and five "
                    "corruption seeds before higher-level inference."
                ),

            "higher_level_units":
                5,

            "higher_level_unit":
                "split_seed",

            "repeated_split_caveat":
                (
                    "The five split seeds are repeated holdout "
                    "sensitivity units, not five independent datasets."
                ),
        },

    "robustness_summary":
        {
            "primary_family_curves":
                [
                    "gaussian",
                    "calibration_offset",
                    "sensor_drift",
                    "sample_dropout",
                ],

            "clean_severity_axis":
                0.0,

            "corrupted_severity_axis":
                [
                    1.0 / 3.0,
                    2.0 / 3.0,
                    1.0,
                ],

            "auc":
                (
                    "Trapezoidal AUC of RMSE and MAE over normalized "
                    "severity axis 0..1, including reconstructed clean "
                    "baseline at severity 0. Lower AUC is better."
                ),

            "per_level_outputs":
                [
                    "absolute metric",
                    "absolute change from clean",
                    "percent change from clean",
                    "MLP-minus-GCN preference margin",
                ],

            "high_severity_output":
                "relative error inflation from clean",

            "compound_medium_role":
                "descriptive only",
        },

    "primary_inferential_sensitivity":
        {
            "endpoint":
                (
                    "AUC over clean+low+medium+high for each primary "
                    "corruption family"
                ),

            "difference_definition":
                (
                    "MLP AUC minus GCN AUC; positive values favor GCN"
                ),

            "families":
                [
                    "gaussian",
                    "calibration_offset",
                    "sensor_drift",
                    "sample_dropout",
                ],

            "domains":
                [
                    "BALANCED_ALL",
                    "HUST",
                    "CALCE",
                    "MATR",
                ],

            "comparisons_per_error_metric":
                16,

            "test":
                (
                    "exact two-sided paired sign-flip sensitivity test "
                    "over five split-level differences"
                ),

            "multiple_testing":
                (
                    "Holm adjustment separately across the 16 RMSE "
                    "AUC comparisons and 16 MAE AUC comparisons"
                ),

            "interpretation":
                (
                    "Inferential sensitivity only; descriptive model "
                    "preference is reported even when Holm-adjusted "
                    "p-values are not significant."
                ),
        },

    "claims_guard":
        {
            "universal_gcn_robustness_claim_allowed":
                False,

            "universal_mlp_robustness_claim_allowed":
                False,

            "preferred_language":
                (
                    "Model preference is corruption-family-, severity-, "
                    "and dataset-dependent unless the frozen results "
                    "support a narrower statement."
                ),
        },

    "source_sha256": {
        name:
            sha256_file(path)
        for name, path in SOURCES.items()
    },

    "protocol_code_commit":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO,
            text=True,
        ).strip(),

    "pre_freeze_physical_test_metric_files":
        0,

    "pre_freeze_physical_evaluation_files":
        0,
}


if PROTOCOL_OUT.exists():

    existing = json.loads(
        PROTOCOL_OUT.read_text(
            encoding="utf-8"
        )
    )

    assert existing == protocol

else:

    atomic_json(
        protocol,
        PROTOCOL_OUT,
    )


print("=" * 80)
print("PHYSICAL SENSOR NOISE PROTOCOL FREEZE")
print("=" * 80)

print(
    "datasets:",
    ",".join(
        protocol[
            "datasets"
        ]
    )
)

print(
    "models:",
    2
)

print(
    "raw corrupted channels:",
    "voltage[V],current[A]"
)

print(
    "temperature primary input:",
    False
)

print(
    "capacity primary input:",
    False
)

print(
    "direct feature-space corruption allowed:",
    False
)

print(
    "clean raw roundtrip required:",
    True
)

print(
    "primary corruption families:",
    4
)

print(
    "levels per primary family:",
    3
)

print(
    "compound medium descriptive:",
    True
)

print(
    "corruption seeds:",
    5
)

print(
    "model init seeds:",
    3
)

print(
    "higher-level split units:",
    5
)

print(
    "Holm family per error metric:",
    16
)

print(
    "physical test metrics before freeze:",
    0
)

print(
    "protocol SHA256:",
    sha256_file(
        PROTOCOL_OUT
    )
)

print()
print(
    "PHYSICAL_SENSOR_NOISE_PROTOCOL_FREEZE_PASS=True"
)
