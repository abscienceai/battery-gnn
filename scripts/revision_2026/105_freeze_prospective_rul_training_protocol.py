from pathlib import Path
import hashlib
import json
import math
import os
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = BASE / "results/revision_2026/prospective_rul"
DATA = ROOT / "data_v1"

MAIN_PROTOCOL = ROOT / "prospective_rul_protocol_v1.json"
DATA_MANIFEST = DATA / "prospective_rul_data_manifest.json"
TARGET = DATA / "prospective_rul_target_table.csv"
ASSIGN = DATA / "inherited_cell_split_assignments.csv"
FEATURE_SCALER = DATA / "train_only_feature_scaler_parameters.csv"
TARGET_SCALER = DATA / "train_only_target_scaler_parameters.csv"

OUT = ROOT / "prospective_rul_training_protocol_v1.json"

SELF = (
    REPO
    / "scripts/revision_2026"
    / "105_freeze_prospective_rul_training_protocol.py"
)


EXPECTED = {
    MAIN_PROTOCOL:
        "2f61fd329008286bca986411a88945c3ac0e5df09aeffab48939bdb92a9f4592",

    DATA_MANIFEST:
        "1bbf306249b658e1233eeba3283e1a1c1044bbfe0aabd43826ace98570845c0b",

    TARGET:
        "e5ec1f09b09f54be06fb9af46bb628b9f81b82053a290ae9c069d1f0d4907725",

    ASSIGN:
        "2b2c5cf7d7e62ecf68879a7df8cfae20e49482b1d36a661de318677272a3a99e",

    FEATURE_SCALER:
        "25f159c51becd400a315aa862d8cdd5f58497172511addff9d7be967314b15aa",

    TARGET_SCALER:
        "78e6e62d3e2d4aa737e43264eabe41823a2d3c5d702151a87569c15e3f2dda8e",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


for path, expected in EXPECTED.items():
    assert path.is_file()
    assert sha256(path) == expected


main = json.loads(
    MAIN_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

data = json.loads(
    DATA_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert main["contract"] == "prospective_rul_protocol_v1"
assert data["contract"] == "prospective_rul_data_v1"

assert data[
    "target_contract"
][
    "total_rows"
] == 64169

assert data[
    "target_contract"
][
    "event_cell_count"
] == 101

assert data[
    "split_contract"
][
    "cells_by_split_per_seed"
] == {
    "train": 71,
    "validation": 15,
    "test": 15,
}


# Exact architecture parameter count:
# 20*64+64 + 64*64+64 + 64*1+1 = 5569
parameter_count = (
    20 * 64 + 64
    + 64 * 64 + 64
    + 64 * 1 + 1
)

assert parameter_count == 5569


rel = str(
    SELF.relative_to(REPO)
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


protocol = {
    "contract":
        "prospective_rul_training_protocol_v1",

    "status":
        "FROZEN_BEFORE_MODEL_TRAINING",

    "task": {
        "target":
            "RUL_observations_80pct_k2",

        "units":
            "observation cycles",

        "datasets": [
            "CALCE",
            "MATR",
        ],

        "HUST_exact_RUL_used":
            False,

        "endpoint":
            (
                "first observation of first run of two consecutive "
                "finite SOH observations <= 0.80"
            ),

        "target_zero_at_EOL":
            True,
    },

    "feature_contract": {
        "feature_count":
            20,

        "feature_scaler":
            "split-specific prospective-RUL TRAIN rows only",

        "target_scaler":
            "split-specific prospective-RUL TRAIN targets only",

        "cycle_index_input":
            False,

        "observation_index_input":
            False,

        "SOH_input":
            False,

        "capacity_input":
            False,

        "RUL_input":
            False,

        "EOL_input":
            False,

        "future_input":
            False,
    },

    "models": {
        "FeatureMatchedMLP_RUL": {
            "layers": [
                "Linear(20,64)",
                "ReLU",
                "Linear(64,64)",
                "ReLU",
                "Linear(64,1)",
            ],

            "learnable_parameters":
                5569,

            "temporal_aggregation":
                False,
        },

        "FeatureMatchedCausalGCN_RUL": {
            "layers": [
                "CausalMeanAggregate",
                "Linear(20,64)",
                "ReLU",
                "CausalMeanAggregate",
                "Linear(64,64)",
                "ReLU",
                "Linear(64,1)",
            ],

            "learnable_parameters":
                5569,

            "aggregation_rule":
                (
                    "for each noninitial valid observation t, "
                    "mean(self_t, previous_valid_observation); "
                    "the first valid observation uses self only"
                ),

            "future_edges":
                False,

            "reverse_edges":
                False,

            "cross_cell_edges":
                False,

            "maximum_receptive_field":
                "t,t-1,t-2",
        },

        "feature_matched":
            True,

        "parameter_matched":
            True,

        "same_initial_learnable_weights_within_pair":
            True,
    },

    "split_and_seed_contract": {
        "split_seeds": [
            0, 1, 2, 3, 4
        ],

        "training_initialization_seeds": [
            0, 1, 2
        ],

        "paired_runs":
            15,

        "total_model_fits":
            30,

        "split_unit":
            "complete cell",

        "reassignment":
            False,

        "test_cell_overlap_with_train_or_validation":
            False,
    },

    "optimization": {
        "device":
            "CPU",

        "training_mode":
            "deterministic full-split full-batch",

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            0.00001,

        "loss_target_space":
            "train-only standardized RUL",

        "loss":
            "hierarchically balanced MSE",

        "training_weight_formula":
            (
                "w(row)=1/N_datasets * "
                "1/N_train_cells_in_dataset * "
                "1/N_train_rows_in_cell; "
                "weights sum to one"
            ),

        "dataset_weighting":
            "equal",

        "cell_weighting_within_dataset":
            "equal",

        "observation_weighting_within_cell":
            "equal",

        "gradient_clipping":
            False,

        "dropout":
            False,

        "batch_normalization":
            False,
    },

    "early_stopping": {
        "selection_data":
            "validation only",

        "selection_metric":
            "BALANCED_ALL validation RMSE in raw RUL observation cycles",

        "target_predictions_inverse_transformed_before_scoring":
            True,

        "patience":
            30,

        "minimum_delta":
            0.0,

        "maximum_epochs_safety_ceiling":
            10000,

        "best_state_retained":
            True,

        "test_metrics_visible_during_selection":
            False,

        "ceiling_policy":
            (
                "if any fit reaches the 10000-epoch safety ceiling, "
                "halt before test evaluation; do not inspect test metrics; "
                "a ceiling-only protocol amendment may then be made using "
                "training/validation evidence only"
            ),
    },

    "initialization": {
        "torch_initialization_seed":
            "training_initialization_seed",

        "same_seed_recreated_immediately_before_each_paired_model":
            True,

        "same_parameter_shapes_and_initial_values_within_pair":
            True,

        "data_shuffle":
            False,
    },

    "validation_metric_semantics": {
        "BALANCED_ALL_RMSE":
            (
                "for each dataset, average per-cell MSE over its "
                "validation cells; average the dataset MSE values equally; "
                "take the square root"
            ),

        "uses_raw_RUL_units":
            True,

        "CALCE_and_MATR_equal_dataset_weight":
            True,
    },

    "test_evaluation": {
        "full_trajectory_role":
            "supportive prospective cycle-level evaluation",

        "fixed_checkpoint_role":
            "primary early prospective evaluation",

        "fixed_checkpoints": [
            25,
            50,
            100,
        ],

        "checkpoint_definition":
            (
                "exact K-th feature-valid observation in the target "
                "table; at most one prediction per cell per checkpoint"
            ),

        "future_lifetime_fraction_used":
            False,

        "domains": [
            "BALANCED_ALL",
            "CALCE",
            "MATR",
        ],

        "metrics": [
            "RMSE",
            "MAE",
            "R2",
            "normalized_MAE",
        ],

        "domain_metric_semantics":
            "plain row-level metrics within CALCE or MATR",

        "balanced_metric_semantics":
            (
                "equal dataset weight, equal cell weight within dataset, "
                "equal observation weight within cell"
            ),

        "normalized_MAE_definition":
            (
                "weighted MAE divided by max(weighted mean absolute "
                "true RUL, 1.0 observation cycle), using the same weights "
                "as the corresponding evaluation domain"
            ),

        "R2_BALANCED_ALL":
            (
                "weighted R2 using the same hierarchical evaluation "
                "weights and weighted target mean"
            ),

        "test_used_for_model_selection":
            False,
    },

    "metric_embargo": {
        "test_predictions_may_be_written_to_artifacts":
            True,

        "test_metrics_may_be_written_to_run_records":
            True,

        "test_numeric_values_printed_by_runner":
            False,

        "test_numeric_values_surfaced_before_independent_execution_audit":
            False,

        "embargo_lift_condition":
            "independent prospective-RUL execution audit passes",
    },

    "analysis_contract": {
        "new_inferential_tests":
            False,

        "primary_reporting_unit":
            (
                "five split-level means; initialization seeds remain "
                "within-split repetitions"
            ),

        "split_seeds_claimed_independent_trials":
            False,

        "model_ranking":
            "descriptive",

        "universal_GCN_superiority_claim_allowed":
            False,

        "old_end_of_record_RUL_results_reused":
            False,

        "old_fractional_early_prediction_results_reused":
            False,
    },

    "frozen_inputs": {
        "main_protocol": {
            "path": str(MAIN_PROTOCOL),
            "sha256": sha256(MAIN_PROTOCOL),
        },

        "data_manifest": {
            "path": str(DATA_MANIFEST),
            "sha256": sha256(DATA_MANIFEST),
        },

        "target_table": {
            "path": str(TARGET),
            "sha256": sha256(TARGET),
        },

        "cell_assignments": {
            "path": str(ASSIGN),
            "sha256": sha256(ASSIGN),
        },

        "feature_scalers": {
            "path": str(FEATURE_SCALER),
            "sha256": sha256(FEATURE_SCALER),
        },

        "target_scalers": {
            "path": str(TARGET_SCALER),
            "sha256": sha256(TARGET_SCALER),
        },
    },

    "protocol_script": {
        "path":
            str(SELF),

        "sha256":
            sha256(SELF),

        "commit":
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO,
                text=True,
            ).strip(),
    },

    "execution_state": {
        "training_started":
            False,

        "model_states_created":
            False,

        "test_metrics_read":
            False,

        "test_metrics_surfaced":
            False,
    },
}


atomic_json(
    protocol,
    OUT,
)


print("PROSPECTIVE RUL TRAINING PROTOCOL V1")
print("models: FeatureMatchedMLP_RUL,FeatureMatchedCausalGCN_RUL")
print("learnable parameters per model: 5569")
print("paired runs: 15")
print("total model fits: 30")
print("loss: dataset/cell/observation-balanced MSE")
print("target training space: train-only standardized RUL")
print("validation selection: balanced raw-RUL RMSE")
print("patience: 30")
print("maximum epoch safety ceiling: 10000")
print("fixed prospective checkpoints: K25,K50,K100")
print("future lifetime fraction used: False")
print("new inferential tests: False")
print("test metric embargo enabled: True")
print("training started: False")
print("test metrics read: False")
print("training_protocol_sha256:", sha256(OUT))
print("PROSPECTIVE_RUL_TRAINING_PROTOCOL_V1_FROZEN=True")
