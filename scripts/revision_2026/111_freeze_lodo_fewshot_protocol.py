from pathlib import Path
import hashlib
import json
import os
import subprocess

import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/lodo_fewshot"
)

OUT = (
    ROOT
    / "lodo_fewshot_protocol_v1.json"
)

TASKS = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

SPLIT_MANIFEST = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "grouped_split_scaler_manifest.json"
)

SELF = (
    REPO
    / "scripts/revision_2026"
    / "111_freeze_lodo_fewshot_protocol.py"
)


DATASETS = [
    "HUST",
    "CALCE",
    "MATR",
]

FEATURES = [
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

SPLIT_SEEDS = [
    0, 1, 2, 3, 4
]

TRAINING_INITIALIZATION_SEEDS = [
    0, 1, 2
]

SUPPORT_CELL_COUNTS = [
    1, 3, 5
]


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


def atomic_json(obj, path):

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
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def boolish(series):

    if series.dtype == bool:
        return series.astype(bool)

    x = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    assert x.notna().all()

    return x.astype(bool)


for p in [
    TASKS,
    ASSIGN,
    SPLIT_MANIFEST,
]:

    assert p.is_file()


assert sha256(
    ASSIGN
) == (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)

assert sha256(
    SPLIT_MANIFEST
) == (
    "91aee27fa332e053454c4f2300c367dc363b938e5ab3d12513372370ec4f2238"
)


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


tasks = pd.read_csv(
    TASKS,
    low_memory=False,
)

assign = pd.read_csv(
    ASSIGN,
    low_memory=False,
)


for c in [
    "feature_valid",
    "soh_task_eligible",
]:

    assert c in tasks.columns

    tasks[c] = boolish(
        tasks[c]
    )


primary = tasks[
    tasks[
        "dataset"
    ].isin(
        DATASETS
    )
    & tasks[
        "feature_valid"
    ]
    & tasks[
        "soh_task_eligible"
    ]
].copy()


assert set(
    primary[
        "dataset"
    ].astype(str)
    .unique()
) == set(
    DATASETS
)


assert [
    c
    for c in FEATURES
    if c not in primary.columns
] == []


assert int(
    (
        primary[
            "dataset"
        ]
        == "HUST"
    ).sum()
) == 146122

assert int(
    (
        primary[
            "dataset"
        ]
        == "CALCE"
    ).sum()
) == 14298

assert int(
    (
        primary[
            "dataset"
        ]
        == "MATR"
    ).sum()
) == 140133


assert primary.loc[
    primary["dataset"] == "HUST",
    "cell_id",
].astype(str).nunique() == 77

assert primary.loc[
    primary["dataset"] == "CALCE",
    "cell_id",
].astype(str).nunique() == 13

assert primary.loc[
    primary["dataset"] == "MATR",
    "cell_id",
].astype(str).nunique() == 169


assign[
    "split_seed"
] = pd.to_numeric(
    assign[
        "split_seed"
    ],
    errors="raise",
).astype(int)


assert set(
    assign[
        "split_seed"
    ].unique()
) == set(
    SPLIT_SEEDS
)


for seed in SPLIT_SEEDS:

    g = assign[
        assign[
            "split_seed"
        ]
        == seed
    ]


    expected = {
        (
            "HUST",
            "train",
        ):
            54,

        (
            "HUST",
            "validation",
        ):
            12,

        (
            "HUST",
            "test",
        ):
            11,

        (
            "CALCE",
            "train",
        ):
            9,

        (
            "CALCE",
            "validation",
        ):
            2,

        (
            "CALCE",
            "test",
        ):
            2,

        (
            "MATR",
            "train",
        ):
            119,

        (
            "MATR",
            "validation",
        ):
            25,

        (
            "MATR",
            "test",
        ):
            25,
    }


    actual = {
        (
            str(k[0]),
            str(k[1]),
        ):
            int(v)
        for k, v in (
            g[
                g[
                    "dataset"
                ].isin(
                    DATASETS
                )
            ]
            .groupby(
                [
                    "dataset",
                    "split",
                ]
            )
            .size()
            .to_dict()
            .items()
        )
    }


    assert actual == expected


folds = {
    held_out: [
        dataset
        for dataset in DATASETS
        if dataset != held_out
    ]
    for held_out in DATASETS
}


protocol = {
    "contract":
        "lodo_fewshot_protocol_v1",

    "status":
        "FROZEN_BEFORE_SUPPORT_SELECTION_SCALER_FIT_OR_MODEL_TRAINING",

    "scientific_scope": {
        "task":
            "cycle-level SOH regression",

        "primary_datasets":
            DATASETS,

        "folds":
            folds,

        "primary_question":
            (
                "How do feature-matched MLP and causal GCN models "
                "generalize to a completely held-out battery dataset, "
                "and how much can a small number of labeled target "
                "training cells improve that transfer?"
            ),

        "old_legacy_lodo_results_reused":
            False,

        "old_legacy_fewshot_results_reused":
            False,
    },

    "input_contract": {
        "canonical_source":
            "canonical_cycle_supervised_table.csv",

        "row_eligibility":
            (
                "dataset in HUST/CALCE/MATR AND "
                "feature_valid=True AND soh_task_eligible=True"
            ),

        "feature_count":
            20,

        "features":
            FEATURES,

        "target":
            "SOH",

        "target_units":
            "dimensionless cell-normalized state of health",

        "dataset_identity_as_model_feature":
            False,

        "cycle_index_as_model_feature":
            False,

        "observation_index_as_model_feature":
            False,

        "capacity_as_model_feature":
            False,

        "RUL_as_model_feature":
            False,

        "EOL_as_model_feature":
            False,
    },

    "lodo_split_contract": {
        "split_seeds":
            SPLIT_SEEDS,

        "frozen_assignment_source":
            "cell_split_assignments.csv",

        "held_out_dataset_removed_from_base_training":
            True,

        "held_out_dataset_removed_from_base_validation":
            True,

        "base_train_cells":
            (
                "train cells from the two source datasets only"
            ),

        "base_validation_cells":
            (
                "validation cells from the two source datasets only"
            ),

        "zero_shot_test_cells":
            (
                "test cells from the held-out target dataset only"
            ),

        "target_train_cells_used_by_zero_shot_model":
            False,

        "target_validation_cells_used_by_zero_shot_model":
            False,

        "target_test_cells_used_for_selection":
            False,

        "cell_overlap_across_train_validation_test":
            False,

        "reassignment":
            False,
    },

    "lodo_scaler_contract": {
        "feature_scaler":
            (
                "StandardScaler-equivalent population mean/variance "
                "fit separately for each held-out dataset and split seed"
            ),

        "fit_rows":
            (
                "feature-valid SOH rows from SOURCE TRAIN cells only"
            ),

        "held_out_target_train_statistics_used":
            False,

        "held_out_target_validation_statistics_used":
            False,

        "held_out_target_test_statistics_used":
            False,

        "source_validation_statistics_used":
            False,

        "same_source_only_scaler_used_for_zero_shot_and_all_few_shot_conditions":
            True,

        "target_support_statistics_refit_scaler":
            False,

        "target_scaler":
            False,
    },

    "zero_shot_model_contract": {
        "models": [
            "FeatureMatchedMLP",
            "FeatureMatchedCausalGCN",
        ],

        "architecture":
            "20 -> 64 -> 64 -> 1 with ReLU",

        "learnable_parameters_each":
            5569,

        "feature_matched":
            True,

        "parameter_matched":
            True,

        "same_initial_learnable_weights_within_pair":
            True,

        "causal_GCN":
            (
                "parameter-free causal mean aggregation over self and "
                "previous valid observation before each shared linear "
                "stage; maximum receptive field t,t-1,t-2"
            ),

        "future_edges":
            False,

        "reverse_edges":
            False,

        "cross_cell_edges":
            False,

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            0.00001,

        "training_initialization_seeds":
            TRAINING_INITIALIZATION_SEEDS,

        "loss":
            "dataset-balanced, cell-balanced, observation-balanced MSE",

        "source_datasets_equal_weight":
            True,

        "source_cells_equal_weight_within_dataset":
            True,

        "rows_equal_weight_within_cell":
            True,

        "early_stopping_data":
            "source validation cells only",

        "early_stopping_metric":
            "source-validation BALANCED_ALL RMSE",

        "patience":
            30,

        "maximum_epochs_safety_ceiling":
            10000,

        "dropout":
            False,

        "batch_normalization":
            False,

        "data_shuffle":
            False,

        "new_hyperparameter_tuning":
            False,
    },

    "few_shot_support_contract": {
        "support_source":
            "held-out target dataset TRAIN cells only",

        "support_cell_counts":
            SUPPORT_CELL_COUNTS,

        "percentage_based_support":
            False,

        "nested_support":
            "K1 subset K3 subset K5",

        "selection_happens_before_model_training":
            True,

        "selection_uses_target_labels":
            False,

        "selection_uses_target_features":
            False,

        "selection_uses_target_validation":
            False,

        "selection_uses_target_test":
            False,

        "selection_rule":
            (
                "deterministically rank eligible target TRAIN cell IDs "
                "by SHA256('lodo_fewshot_v1|held_out=<dataset>|"
                "split_seed=<seed>|cell_id=<cell>'); "
                "take first 1, 3, or 5 cells"
            ),

        "same_support_cells_for_MLP_and_GCN":
            True,

        "same_support_cells_across_training_initialization_seeds":
            True,

        "exact_support_cell_ids_materialized":
            True,

        "exact_support_row_counts_materialized":
            True,

        "target_validation_cells_reserved_and_never_used":
            True,

        "target_test_cells_reserved_for_final_evaluation_only":
            True,
    },

    "few_shot_adaptation_contract": {
        "starting_state":
            "independently from the corresponding zero-shot best state",

        "K3_starts_from_K1_state":
            False,

        "K5_starts_from_K3_state":
            False,

        "all_layers_trainable":
            True,

        "optimizer":
            "AdamW",

        "learning_rate":
            0.0001,

        "learning_rate_rule":
            "one tenth of frozen zero-shot learning rate",

        "weight_decay":
            0.00001,

        "epochs":
            100,

        "training_mode":
            "deterministic full-support full-batch",

        "loss":
            "cell-balanced and observation-balanced target-support MSE",

        "early_stopping":
            False,

        "target_validation_used_for_epoch_selection":
            False,

        "target_test_used_during_adaptation":
            False,

        "feature_scaler_refit_during_adaptation":
            False,

        "new_adaptation_hyperparameter_tuning":
            False,
    },

    "evaluation_contract": {
        "conditions": [
            "ZERO_SHOT",
            "K1",
            "K3",
            "K5",
        ],

        "evaluation_set":
            (
                "the same frozen held-out target TEST cells for "
                "ZERO_SHOT, K1, K3, and K5"
            ),

        "primary_metrics": [
            "CELL_BALANCED_RMSE",
            "CELL_BALANCED_MAE",
        ],

        "secondary_metrics": [
            "CELL_BALANCED_R2",
            "ROW_RMSE",
            "ROW_MAE",
            "ROW_R2",
        ],

        "cell_balanced_semantics":
            (
                "each held-out test cell receives equal total weight; "
                "rows within a cell receive equal weight"
            ),

        "row_metric_semantics":
            "plain unweighted row-level metric",

        "SOH_scale":
            "dimensionless and cell-normalized before modeling",

        "R2_interpretation":
            (
                "secondary only because target-domain SOH variance can "
                "differ strongly across datasets; especially narrow "
                "target variance can make R2 unstable despite finite "
                "absolute error"
            ),

        "relative_fewshot_improvement":
            (
                "paired within identical held-out dataset, split seed, "
                "training initialization, model, and test set"
            ),

        "test_used_for_model_selection":
            False,
    },

    "reporting_contract": {
        "folds":
            DATASETS,

        "five_split_seeds_are_independent_trials":
            False,

        "three_initialization_seeds_are_independent_splits":
            False,

        "primary_summary":
            (
                "mean over 15 split/init runs per fold/model/condition; "
                "uncertainty descriptor is SD over the five split-level "
                "means after averaging three initialization repetitions"
            ),

        "new_inferential_tests":
            False,

        "model_ranking":
            "descriptive",

        "few_shot_claim":
            (
                "adaptation benefit is empirical and dataset-dependent; "
                "no monotonic or universal improvement claim is assumed"
            ),

        "universal_GCN_superiority_claim":
            False,

        "failure_is_reportable_result":
            True,
    },

    "metric_embargo": {
        "test_predictions_may_be_written":
            True,

        "test_metrics_may_be_written_to_run_records":
            True,

        "numeric_test_values_printed_by_runner":
            False,

        "numeric_test_values_surfaced_before_execution_audit":
            False,

        "lift_condition":
            "independent LODO/few-shot execution audit passes",
    },

    "expected_execution_scale": {
        "zero_shot_folds":
            3,

        "split_seeds":
            5,

        "training_initializations":
            3,

        "models":
            2,

        "zero_shot_model_fits":
            90,

        "few_shot_conditions_per_zero_shot_state":
            3,

        "few_shot_adaptation_fits":
            270,

        "total_fits_or_adaptations":
            360,
    },

    "frozen_observed_inventory": {
        "rows": {
            "HUST":
                146122,

            "CALCE":
                14298,

            "MATR":
                140133,
        },

        "cells": {
            "HUST":
                77,

            "CALCE":
                13,

            "MATR":
                169,
        },

        "target_train_cell_pool_size_per_split": {
            "HUST":
                54,

            "CALCE":
                9,

            "MATR":
                119,
        },
    },

    "source_artifacts": {
        "canonical_cycle_supervised_table": {
            "path":
                str(TASKS),

            "sha256":
                sha256(TASKS),
        },

        "cell_split_assignments": {
            "path":
                str(ASSIGN),

            "sha256":
                sha256(ASSIGN),
        },

        "grouped_split_scaler_manifest": {
            "path":
                str(SPLIT_MANIFEST),

            "sha256":
                sha256(SPLIT_MANIFEST),
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
        "support_cells_selected":
            False,

        "lodo_scalers_fit":
            False,

        "zero_shot_training_started":
            False,

        "few_shot_adaptation_started":
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


print("LODO / FEW-SHOT PROTOCOL V1")
print("folds: HUST,CALCE,MATR")
print("features: 20")
print("models: feature-matched MLP,causal GCN")
print("parameters per model: 5569")
print("target data used for zero-shot scaler: False")
print("target train data used for zero-shot training: False")
print("target validation used anywhere: False")
print("support sizes: K1,K3,K5 target-train cells")
print("support selection deterministic: True")
print("support sets nested: True")
print("adaptation epochs: 100")
print("adaptation learning rate: 0.0001")
print("target validation adaptation selection: False")
print("target test adaptation selection: False")
print("source-only scaler retained during adaptation: True")
print("new inferential tests: False")
print("test metric embargo enabled: True")
print("support selection started: False")
print("training started: False")
print("protocol_sha256:", sha256(OUT))
print("LODO_FEWSHOT_PROTOCOL_V1_FROZEN=True")
