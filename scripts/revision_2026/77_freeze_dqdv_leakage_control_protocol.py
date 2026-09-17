from pathlib import Path
import hashlib
import json
import os
import subprocess


BASE = Path.home() / "alper" / "battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

OUT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

PROTOCOL = OUT / "protocol.json"

SELF = (
    REPO
    / "scripts/revision_2026/77_freeze_dqdv_leakage_control_protocol.py"
)

AUDIT = (
    REPO
    / "scripts/revision_2026/78_dqdv_leakage_control_protocol_audit.py"
)

RAW_GENERATOR = (
    REPO
    / "scripts/revision_2026/14_raw_cycle_feature_generator.py"
)

FEATURE_KERNEL = (
    REPO
    / "scripts/revision_2026/cycle_feature_kernel.py"
)

HUST_AUDIT = (
    REPO
    / "scripts/revision_2026/00_hust_dataset_audit.py"
)

FINAL_V3_PROTOCOL = (
    BASE
    / "results/revision_2026/feature_matched_soh"
    / "full_clean_benchmark_protocol_v3.json"
)

EXPECTED_FINAL_V3_PROTOCOL_SHA = (
    "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd"
)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def atomic_json(obj, path):
    tmp = Path(
        str(path) + ".tmp"
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


assert FINAL_V3_PROTOCOL.is_file()

assert sha256(
    FINAL_V3_PROTOCOL
) == EXPECTED_FINAL_V3_PROTOCOL_SHA


for path in [
    SELF,
    AUDIT,
    RAW_GENERATOR,
    FEATURE_KERNEL,
    HUST_AUDIT,
]:
    assert path.is_file()


head = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
    text=True,
).strip()


for path in [
    SELF,
    AUDIT,
]:
    rel = str(
        path.relative_to(
            REPO
        )
    )

    subprocess.check_call(
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

    assert subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            rel,
        ],
        cwd=REPO,
    ).returncode == 0


if PROTOCOL.exists():
    raise RuntimeError(
        "Protocol already exists; refusing overwrite."
    )


dqdv_features = [
    "ica_mean_Ah_per_V",
    "ica_std_Ah_per_V",
    "ica_q10_Ah_per_V",
    "ica_q25_Ah_per_V",
    "ica_median_Ah_per_V",
    "ica_q75_Ah_per_V",
    "ica_q90_Ah_per_V",
    "ica_max_Ah_per_V",
    "ica_area_Ah",
    "ica_peak_voltage_V",
]


conditions = [
    {
        "id":
            "PRIMARY20_COMMON",

        "input_dimension":
            20,

        "feature_set":
            "primary 20 V/I descriptors only",

        "role":
            "common-cohort reference",
    },

    {
        "id":
            "PRIMARY20_PLUS_FULL_DQDV10",

        "input_dimension":
            30,

        "feature_set":
            "primary20 + full-curve dQ/dV10",

        "role":
            "retrospective leakage-positive control",
    },

    {
        "id":
            "PRIMARY20_PLUS_PREFIX25_DQDV10",

        "input_dimension":
            30,

        "feature_set":
            "primary20 + causal fixed-time prefix dQ/dV10",

        "role":
            "causal truncated leakage control",
    },

    {
        "id":
            "PRIMARY20_PLUS_PREFIX50_DQDV10",

        "input_dimension":
            30,

        "feature_set":
            "primary20 + causal fixed-time prefix dQ/dV10",

        "role":
            "causal truncated leakage control",
    },
]


protocol = {
    "contract":
        "dqdv_leakage_control_protocol_v1",

    "status":
        "FROZEN_BEFORE_DQDV_FEATURE_EXTRACTION",

    "freeze_commit":
        head,

    "source_anchors": {
        "final_v3_protocol": {
            "path":
                str(
                    FINAL_V3_PROTOCOL
                ),

            "sha256":
                sha256(
                    FINAL_V3_PROTOCOL
                ),
        },

        "raw_cycle_feature_generator": {
            "path":
                str(
                    RAW_GENERATOR
                ),

            "sha256":
                sha256(
                    RAW_GENERATOR
                ),
        },

        "cycle_feature_kernel": {
            "path":
                str(
                    FEATURE_KERNEL
                ),

            "sha256":
                sha256(
                    FEATURE_KERNEL
                ),
        },

        "hust_dataset_audit": {
            "path":
                str(
                    HUST_AUDIT
                ),

            "sha256":
                sha256(
                    HUST_AUDIT
                ),
        },

        "freeze_script": {
            "path":
                str(
                    SELF
                ),

            "sha256":
                sha256(
                    SELF
                ),
        },

        "audit_script": {
            "path":
                str(
                    AUDIT
                ),

            "sha256":
                sha256(
                    AUDIT
                ),
        },
    },

    "task": {
        "name":
            "cycle-level SOH leakage sensitivity control",

        "datasets": [
            "HUST",
            "CALCE",
            "MATR",
        ],

        "split_seeds": [
            0,
            1,
            2,
            3,
            4,
        ],

        "train_initialization_seeds": [
            0,
            1,
            2,
        ],

        "reuse_exact_final_v3_group_assignments":
            True,

        "cycle_index_is_model_feature":
            False,

        "capacity_target_is_model_feature":
            False,

        "SOH_target_is_model_feature":
            False,

        "RUL_or_EOL_is_model_feature":
            False,
    },

    "raw_trace_contract": {
        "only_sensor_inputs": [
            "voltage_V",
            "current_A",
            "time_s",
        ],

        "target_tables_forbidden_during_dqdv_generation": [
            "capacity_Ah",
            "SOH",
            "RUL",
            "EOL",
            "HUST source dq",
            "MATR summary QDischarge",
            "MATR summary QCharge",
            "CALCE capacity columns",
        ],

        "dataset_semantics": {
            "HUST": {
                "voltage":
                    "Voltage (V)",

                "current":
                    "Current (mA) converted to A",

                "time":
                    "Time (s)",

                "discharge_rule":
                    "I_A < 0",
            },

            "CALCE": {
                "voltage":
                    "Voltage(V)",

                "current":
                    "Current(A)",

                "time":
                    "Test_Time(s)",

                "discharge_rule":
                    "I_A < 0",
            },

            "MATR": {
                "voltage":
                    "cycles.V",

                "current":
                    "cycles.I multiplied by 1.1 A",

                "time":
                    "cycles.t converted from minutes to seconds",

                "discharge_rule":
                    "I_A < 0",
            },
        },

        "polarity_validation_barrier_required_before_bulk_generation":
            True,

        "polarity_validation_rule":
            (
                "negative-current full-discharge integration must "
                "be consistent with each dataset's already-audited "
                "discharge-capacity provenance before model features "
                "are generated; failure stops execution"
            ),
    },

    "incremental_capacity_definition": {
        "discharge_onset":
            (
                "start timestamp of the first valid interval whose "
                "end-sample current is negative"
            ),

        "interval_current_rule":
            (
                "BatteryML-compatible right-endpoint current: "
                "I_step = I[1:]"
            ),

        "valid_interval":
            (
                "finite endpoints, finite current, dt > 0, "
                "I_step < 0"
            ),

        "interval_charge_Ah":
            "dq_Ah = (-I_step) * dt_s / 3600",

        "voltage_representative":
            (
                "midpoint voltage of the contributing interval; "
                "for a prefix-crossing interval the endpoint voltage "
                "is linearly interpolated to the cutoff before taking "
                "the midpoint"
            ),

        "voltage_bin_origin_V":
            0.0,

        "voltage_bin_width_V":
            0.01,

        "bin_assignment":
            (
                "floor((V_mid - 0.0)/0.01); "
                "all dq in one interval is accumulated into that bin"
            ),

        "ica_bin_value":
            "sum(dq_Ah_in_bin) / 0.01 V",

        "feature_statistics_use":
            "occupied positive-charge bins only",

        "ica_area_definition":
            (
                "sum(ica_bin * 0.01 V), numerically equal to "
                "observed cumulative discharge Ah"
            ),

        "peak_voltage_tie_break":
            "highest voltage bin center",

        "minimum_discharge_intervals":
            20,

        "minimum_occupied_voltage_bins":
            5,

        "feature_names":
            dqdv_features,

        "feature_count":
            10,
    },

    "information_horizons": {
        "FULL": {
            "role":
                "retrospective leakage-positive control",

            "cutoff":
                "last available discharge-contributing interval",

            "deployment_valid":
                False,

            "why_leakage_prone":
                (
                    "ica_area_Ah over the full discharge trace is "
                    "the full observed discharged charge and therefore "
                    "can encode the capacity target directly or nearly "
                    "directly"
                ),
        },

        "PREFIX25": {
            "role":
                "causal truncated control",

            "cutoff_fit_scope":
                "training rows only, separately for dataset x split_seed",

            "cutoff_formula":
                (
                    "0.25 * median(training full-discharge wall-clock "
                    "duration from first to last discharge-contributing "
                    "interval)"
                ),

            "validation_or_test_duration_used_to_fit_cutoff":
                False,

            "individual_test_cycle_endpoint_used":
                False,

            "crossing_interval_policy":
                "clip exactly at fixed elapsed-time cutoff",

            "deployment_valid":
                True,
        },

        "PREFIX50": {
            "role":
                "causal truncated control",

            "cutoff_fit_scope":
                "training rows only, separately for dataset x split_seed",

            "cutoff_formula":
                (
                    "0.50 * median(training full-discharge wall-clock "
                    "duration from first to last discharge-contributing "
                    "interval)"
                ),

            "validation_or_test_duration_used_to_fit_cutoff":
                False,

            "individual_test_cycle_endpoint_used":
                False,

            "crossing_interval_policy":
                "clip exactly at fixed elapsed-time cutoff",

            "deployment_valid":
                True,
        },
    },

    "cohort_policy": {
        "per_split_common_cohort":
            True,

        "definition":
            (
                "within each split_seed and partition, retain only "
                "canonical SOH rows that have valid primary20 features "
                "and valid FULL, PREFIX25, and PREFIX50 dQ/dV10; use "
                "the identical retained row IDs for all four conditions"
            ),

        "condition_specific_test_cohorts_forbidden":
            True,

        "purpose":
            "prevent row-availability differences from confounding horizon comparisons",
    },

    "feature_sets": {
        "conditions":
            conditions,

        "same_dqdv_feature_dimension_across_full_and_prefix_conditions":
            True,

        "primary20_definition_unchanged_from_final_v3":
            True,
    },

    "scaling": {
        "fit_scope":
            "training rows only within each split's common cohort",

        "primary20_scaler":
            (
                "fit once per split on common training rows and reused "
                "identically across all four conditions"
            ),

        "dqdv10_scaler":
            (
                "fit separately for FULL, PREFIX25, PREFIX50 using "
                "only common training rows"
            ),

        "validation_or_test_statistics_used":
            False,
    },

    "models": {
        "architectures": [
            "FeatureMatchedMLP",
            "FeatureMatchedCausalGCN",
        ],

        "hidden_layers":
            [
                64,
                64,
            ],

        "activation":
            "ReLU",

        "dropout":
            False,

        "batch_norm":
            False,

        "causal_gcn_operator":
            (
                "same parameter-free mean aggregation over "
                "{self, previous valid observation} used in final-v3"
            ),

        "future_edges":
            False,

        "cross_cell_edges":
            False,

        "parameters_primary20":
            5569,

        "parameters_augmented30":
            6209,

        "mlp_and_gcn_parameter_matched_within_each_condition":
            True,

        "retrain_required":
            True,

        "frozen_final_v3_states_reused":
            False,
    },

    "optimization": {
        "reuse_final_v3_optimizer_objective_and_validation_logic":
            True,

        "objective":
            "dataset-balanced + cell-balanced MSE",

        "early_stopping_metric":
            "validation balanced RMSE",

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            0.00001,

        "test_set_used_for_training_or_selection":
            False,

        "new_hyperparameter_tuning":
            False,
    },

    "execution_size": {
        "conditions":
            4,

        "split_seeds":
            5,

        "train_initializations":
            3,

        "models":
            2,

        "expected_model_fits":
            120,
    },

    "analysis_contract": {
        "role":
            "target-leakage sensitivity and causal truncation control",

        "primary_metrics": [
            "RMSE",
            "MAE",
        ],

        "domains": [
            "BALANCED_ALL",
            "HUST",
            "CALCE",
            "MATR",
        ],

        "full_curve_capacity_alignment_report": [
            "MAE between ica_area_Ah and canonical capacity_Ah",
            "median absolute relative error",
            "R2",
        ],

        "model_performance_reports": [
            "PRIMARY20_COMMON",
            "PRIMARY20_PLUS_FULL_DQDV10",
            "PRIMARY20_PLUS_PREFIX25_DQDV10",
            "PRIMARY20_PLUS_PREFIX50_DQDV10",
        ],

        "descriptive_horizon_contrasts": [
            "FULL minus PREFIX50",
            "FULL minus PREFIX25",
            "PREFIX50 minus PRIMARY20_COMMON",
            "PREFIX25 minus PRIMARY20_COMMON",
        ],

        "new_hypothesis_tests":
            False,

        "p_values":
            False,

        "inference_claim":
            "descriptive leakage-control evidence only",
    },

    "claim_guards": {
        "full_curve_dqdv_is_deployment_valid":
            False,

        "full_curve_result_can_support_prospective_claim":
            False,

        "prefix_cutoff_can_use_test_cycle_total_duration":
            False,

        "prefix_cutoff_can_use_test_cycle_endpoint":
            False,

        "target_capacity_can_enter_feature_generator":
            False,

        "full_curve_dqdv_is_positive_control_for_target_leakage":
            True,

        "causal_prefix_dqdv_is_target_independent_at_generation_time":
            True,

        "feature_dimension_difference_vs_primary20_must_be_disclosed":
            True,

        "universal_gcn_advantage_claim":
            False,
    },

    "freeze_barriers": {
        "raw_numeric_trace_values_read":
            False,

        "capacity_values_read":
            False,

        "SOH_values_read":
            False,

        "model_metrics_read":
            False,

        "model_states_loaded":
            False,

        "dqdv_features_computed":
            False,

        "model_training_run":
            False,

        "analysis_performed":
            False,
    },
}


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_json(
    protocol,
    PROTOCOL,
)


print(
    "DQDV LEAKAGE-CONTROL PROTOCOL FROZEN"
)

print(
    "datasets=HUST,CALCE,MATR"
)

print(
    "information_horizons=FULL,PREFIX25,PREFIX50"
)

print(
    "prefix_cutoff_source=train_only_dataset_x_split_median_duration"
)

print(
    "test_cycle_total_duration_used=False"
)

print(
    "full_curve_role=retrospective_leakage_positive_control"
)

print(
    "dqdv_feature_count=10"
)

print(
    "common_cohort=True"
)

print(
    "conditions=4"
)

print(
    "expected_model_fits=120"
)

print(
    "new_hypothesis_tests=False"
)

print(
    "raw_numeric_trace_values_read=False"
)

print(
    "capacity_values_read=False"
)

print(
    "model_metrics_read=False"
)

print(
    "protocol_sha256=",
    sha256(
        PROTOCOL
    ),
)

print(
    "DQDV_LEAKAGE_CONTROL_PROTOCOL_FREEZE_PASS=True"
)
