from pathlib import Path
import hashlib
import json
import os
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

OUTDIR = (
    BASE
    / "results/revision_2026/prospective_rul"
)

OUT = (
    OUTDIR
    / "prospective_rul_protocol_v1.json"
)

SELF = (
    REPO
    / "scripts/revision_2026/101_freeze_prospective_rul_protocol.py"
)

LABELS = (
    BASE
    / "data/processed/revision_2026/unified_label_backbone.csv"
)

TASKS = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

FEATURE_KERNEL = (
    REPO
    / "scripts/revision_2026/cycle_feature_kernel.py"
)


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def atomic_json(
    obj,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


for path in [
    LABELS,
    TASKS,
    FEATURE_KERNEL,
]:
    assert path.is_file()


rel = str(
    SELF.relative_to(
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


protocol = {
    "contract":
        "prospective_rul_protocol_v1",

    "status":
        "FROZEN_BEFORE_TARGET_BUILDING_OR_MODEL_TRAINING",

    "scientific_scope": {
        "task":
            (
                "cycle-level prospective remaining-useful-life "
                "regression to a harmonized observed 80%-SOH EOL"
            ),

        "primary_datasets": [
            "CALCE",
            "MATR",
        ],

        "hust_role":
            (
                "threshold-censored external diagnostic only; "
                "no fabricated exact 80%-SOH RUL labels"
            ),

        "rwth_role":
            "not part of this exact-RUL benchmark",
    },

    "endpoint_contract": {
        "primary_definition":
            (
                "EOL is the first observation of the first run of "
                "two consecutive finite SOH observations <= 0.80"
            ),

        "threshold":
            0.80,

        "required_consecutive_observations":
            2,

        "consecutive_observation_index_required":
            True,

        "rul_definition":
            (
                "RUL_observations_80pct_k2 = "
                "harmonized_eol_observation_index - observation_index"
            ),

        "rul_at_eol":
            0,

        "post_eol_rows_receive_rul":
            False,

        "right_censored_cells_receive_exact_rul":
            False,

        "sensitivity_endpoints": {
            "k1":
                (
                    "first finite SOH <= 0.80; sensitivity only"
                ),

            "k3":
                (
                    "first observation of first three-consecutive "
                    "finite SOH <= 0.80 run; sensitivity only"
                ),
        },
    },

    "feasibility_lock": {
        "HUST": {
            "cells":
                77,

            "primary_k2_event_cells":
                0,

            "minimum_cell_min_soh":
                0.800005,

            "interpretation":
                (
                    "no observed 80%-SOH endpoint; exact labels forbidden"
                ),
        },

        "CALCE": {
            "cells":
                13,

            "k1_event_cells":
                13,

            "k2_event_cells":
                13,

            "k3_event_cells":
                13,

            "primary_k2_exact_rul_rows":
                7355,

            "primary_k2_eol_rows":
                13,

            "k2_matches_existing_canonical_event_set":
                True,

            "k2_matches_existing_canonical_eol_index":
                True,
        },

        "MATR": {
            "cells":
                169,

            "k1_event_cells":
                129,

            "k2_event_cells":
                88,

            "k3_event_cells":
                88,

            "k1_only_cells":
                41,

            "k1_only_crossing_rows_feature_valid":
                0,

            "primary_k2_exact_rul_rows":
                56814,

            "primary_k2_eol_rows":
                88,

            "k2_matches_existing_observed_event_cell_set":
                True,

            "k2_matches_existing_canonical_eol_index":
                False,

            "median_abs_k2_vs_existing_eol_index_difference":
                16.0,

            "max_abs_k2_vs_existing_eol_index_difference":
                48,
        },
    },

    "input_contract": {
        "feature_source":
            (
                "canonical_cycle_supervised_table.csv"
            ),

        "input_eligibility":
            "feature_valid",

        "existing_rul_task_eligible_used":
            False,

        "feature_count":
            20,

        "feature_family":
            (
                "within-cycle voltage/current statistics only"
            ),

        "cycle_index_as_model_input":
            False,

        "observation_index_as_model_input":
            False,

        "SOH_as_model_input":
            False,

        "capacity_as_model_input":
            False,

        "RUL_as_model_input":
            False,

        "EOL_as_model_input":
            False,

        "lifetime_as_model_input":
            False,

        "future_observation_as_model_input":
            False,
    },

    "prospective_context_contract": {
        "prediction_at_observation_t":
            (
                "may use only feature-valid observations with "
                "observation time/index <= t"
            ),

        "future_nodes_visible":
            False,

        "reverse_future_edges":
            False,

        "cross_cell_edges":
            False,

        "causal_gcn_context":
            (
                "self plus previous valid observations only; "
                "two message-passing stages imply maximum historical "
                "receptive field t,t-1,t-2"
            ),

        "retrospective_global_pooling":
            False,
    },

    "early_prediction_contract": {
        "uses_fraction_of_future_lifetime":
            False,

        "primary_checkpoints_observed_cycles": [
            25,
            50,
            100,
        ],

        "checkpoint_definition":
            (
                "exact K-th feature-valid observation available so far; "
                "one prediction per eligible cell at each checkpoint"
            ),

        "future_lifetime_needed_to_define_checkpoint":
            False,
    },

    "split_contract": {
        "unit":
            "complete canonical cell",

        "split_seeds": [
            0,
            1,
            2,
            3,
            4,
        ],

        "train_validation_test_policy":
            (
                "reuse the already frozen grouped full-cell split "
                "assignments; subset to CALCE/MATR k2 event cells only"
            ),

        "row_level_random_split":
            False,

        "cell_overlap_across_splits":
            False,

        "test_data_used_for_model_selection":
            False,
    },

    "scaling_contract": {
        "feature_scalers":
            (
                "fit anew on prospective-RUL TRAIN rows only for each "
                "split seed; do not inherit the SOH-task pooled scaler"
            ),

        "target_scaler":
            (
                "fit on prospective-RUL TRAIN targets only; "
                "training transformation must be invertible for "
                "raw-observation-cycle evaluation"
            ),

        "validation_or_test_rows_used_to_fit_scalers":
            False,
    },

    "model_contract": {
        "primary_models": [
            "FeatureMatchedMLP_RUL",
            "FeatureMatchedCausalGCN_RUL",
        ],

        "same_input_features":
            True,

        "same_split_assignments":
            True,

        "same_target_definition":
            True,

        "same_training_objective":
            True,

        "same_initial_learnable_weights_within_pair":
            True,

        "gcn_message_passing":
            (
                "parameter-free causal mean aggregation followed by "
                "shared learnable linear layers; no future aggregation"
            ),

        "graph_level_lifetime_regression":
            False,

        "node_or_cycle_level_prediction":
            True,
    },

    "training_contract": {
        "training_initialization_seeds": [
            0,
            1,
            2,
        ],

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            0.00001,

        "early_stopping_metric":
            (
                "validation-only balanced raw-RUL RMSE"
            ),

        "patience":
            30,

        "test_metrics_available_during_training":
            False,

        "new_hyperparameter_tuning":
            False,
    },

    "evaluation_contract": {
        "primary_metrics": [
            "RMSE_observations",
            "MAE_observations",
        ],

        "secondary_metrics": [
            "R2",
            "normalized_MAE",
        ],

        "primary_domains": [
            "BALANCED_ALL",
            "CALCE",
            "MATR",
        ],

        "balanced_semantics":
            (
                "dataset-balanced then cell-balanced so a long-life "
                "dataset/cell cannot dominate by row count"
            ),

        "domain_semantics":
            (
                "plain row-level metrics within each dataset, "
                "reported alongside balanced metrics"
            ),

        "early_checkpoint_metrics":
            [
                "K25",
                "K50",
                "K100",
            ],

        "five_split_seeds_are_independent_trials":
            False,

        "three_initialization_seeds_within_split_are_independent_splits":
            False,
    },

    "interpretation_barriers": {
        "claim_universal_gcn_superiority":
            False,

        "claim_HUST_80pct_RUL_performance":
            False,

        "claim_censored_exact_RUL":
            False,

        "claim_future_lifetime_fraction_is_prospective":
            False,

        "old_end_of_record_RUL_results_reused":
            False,

        "old_early_prediction_results_reused":
            False,

        "primary_claim":
            (
                "prospective causal benchmark under a harmonized "
                "observed endpoint, with model ranking treated as empirical"
            ),
    },

    "source_artifacts": {
        "unified_label_backbone": {
            "path":
                str(
                    LABELS
                ),

            "sha256":
                sha256(
                    LABELS
                ),
        },

        "canonical_cycle_supervised_table": {
            "path":
                str(
                    TASKS
                ),

            "sha256":
                sha256(
                    TASKS
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
    },

    "protocol_script": {
        "path":
            str(
                SELF
            ),

        "sha256":
            sha256(
                SELF
            ),

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
        "target_table_built":
            False,

        "scalers_fit":
            False,

        "model_training_performed":
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


print("PROSPECTIVE RUL PROTOCOL V1")
print(
    "primary datasets: CALCE,MATR"
)
print(
    "HUST exact 80pct RUL excluded: True"
)
print(
    "primary endpoint: first of 2 consecutive SOH<=0.80"
)
print(
    "CALCE event cells: 13"
)
print(
    "MATR event cells: 88"
)
print(
    "CALCE exact RUL rows: 7355"
)
print(
    "MATR exact RUL rows: 56814"
)
print(
    "fixed prospective checkpoints: K25,K50,K100"
)
print(
    "future lifetime fraction used: False"
)
print(
    "retrospective global pooling: False"
)
print(
    "existing rul_task_eligible reused: False"
)
print(
    "model training performed: False"
)
print(
    "test metrics read: False"
)
print(
    "protocol_sha256:",
    sha256(
        OUT
    )
)
print(
    "PROSPECTIVE_RUL_PROTOCOL_V1_FROZEN=True"
)
