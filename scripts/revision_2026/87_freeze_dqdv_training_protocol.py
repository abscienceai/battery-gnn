from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

DQROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

OUT = DQROOT / "training_v1"

DQ_PROTOCOL = DQROOT / "protocol.json"

COMMON_MANIFEST = (
    DQROOT
    / "common_cohort_scalers_v1/common_cohort_scaler_manifest.json"
)

FEATURE_BANK_MANIFEST = (
    DQROOT
    / "features_v1/dqdv_feature_bank_manifest.json"
)

FINAL_V3_PROTOCOL = (
    BASE
    / "results/revision_2026/feature_matched_soh"
    / "full_clean_benchmark_protocol_v3.json"
)

FINAL_V3_MANIFEST = (
    BASE
    / "results/revision_2026/feature_matched_soh/final_v3"
    / "final_v3_execution_manifest.json"
)

MODEL_IMPL = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

FINAL_V3_RUNNER = (
    REPO
    / "scripts/revision_2026/29_full_clean_soh_v3_runner.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/87_freeze_dqdv_training_protocol.py"
)

PROTOCOL_OUT = OUT / "dqdv_training_protocol_v1.json"


EXPECTED = {
    "dqdv_protocol":
        "c455bf7a46f89649f25769a43abd443f171545b06e4740c799df1842c8bbb8e9",

    "common_manifest":
        "3ee48af31ad580c14de0460fa1f163ad572e369368d6da71c0ba6823d28905a1",

    "feature_bank_manifest":
        "901d692f720a9e5166283f66b462580496192e9bab8cc1706453546c167021dc",

    "final_v3_protocol":
        "2dec14c210ae3f02e5f65e4acc24f9c4adc747bc071509cb144e69dba707f4dd",

    "final_v3_manifest":
        "5a8bfbe0f11a6158f2749a9da4f18f1e62adf8c839976c1bf0af9c9f6c07440d",

    "final_v3_runner":
        "b945728c02255f1f2977e47024e33ca75f4e76570c8d381b4881836f17d3dcba",
}


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


# ------------------------------------------------------------
# Exact immutable upstream anchors.
# ------------------------------------------------------------

assert sha256(
    DQ_PROTOCOL
) == EXPECTED[
    "dqdv_protocol"
]

assert sha256(
    COMMON_MANIFEST
) == EXPECTED[
    "common_manifest"
]

assert sha256(
    FEATURE_BANK_MANIFEST
) == EXPECTED[
    "feature_bank_manifest"
]

assert sha256(
    FINAL_V3_PROTOCOL
) == EXPECTED[
    "final_v3_protocol"
]

assert sha256(
    FINAL_V3_MANIFEST
) == EXPECTED[
    "final_v3_manifest"
]

assert sha256(
    FINAL_V3_RUNNER
) == EXPECTED[
    "final_v3_runner"
]


# ------------------------------------------------------------
# Scripts used as implementation anchors must be tracked + clean.
# ------------------------------------------------------------

for path in [
    MODEL_IMPL,
    FINAL_V3_RUNNER,
    SELF,
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


# ------------------------------------------------------------
# Confirm model implementation API exists.
# ------------------------------------------------------------

tree = ast.parse(
    MODEL_IMPL.read_text(
        encoding="utf-8"
    )
)

classes = {
    n.name
    for n in tree.body
    if isinstance(
        n,
        ast.ClassDef,
    )
}

functions = {
    n.name
    for n in tree.body
    if isinstance(
        n,
        ast.FunctionDef,
    )
}

assert {
    "FeatureMatchedMLP",
    "FeatureMatchedCausalGCN",
}.issubset(
    classes
)

assert {
    "set_seed",
    "weighted_metrics",
    "dataset_metrics",
    "train_one",
}.issubset(
    functions
)


# ------------------------------------------------------------
# Read only contract-level fields from final-v3 artifacts.
# Never inspect result records or test metric values.
# ------------------------------------------------------------

v3p = json.loads(
    FINAL_V3_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

v3m = json.loads(
    FINAL_V3_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert v3p[
    "architecture"
] == "20 -> 64 -> 64 -> 1"

assert int(
    v3p[
        "hidden_dim"
    ]
) == 64

assert float(
    v3p[
        "learning_rate"
    ]
) == 0.001

assert float(
    v3p[
        "weight_decay"
    ]
) == 1e-5

assert str(
    v3p[
        "optimizer"
    ]
) == "AdamW"

assert int(
    v3p[
        "max_epochs"
    ]
) == 5000

assert int(
    v3p[
        "early_stopping_patience"
    ]
) == 30

assert str(
    v3p[
        "objective"
    ]
) == "dataset-balanced + cell-balanced MSE"

assert str(
    v3p[
        "early_stopping_metric"
    ]
) == "validation dataset-balanced cell-balanced RMSE"

assert list(
    v3p[
        "split_seeds"
    ]
) == [
    0, 1, 2, 3, 4
]

assert list(
    v3p[
        "training_initialization_seeds"
    ]
) == [
    0, 1, 2
]

assert int(
    v3m[
        "epoch_ceiling"
    ]
) == 5000

assert int(
    v3m[
        "patience"
    ]
) == 30

assert v3m[
    "all_fits_below_epoch_ceiling"
] is True

assert v3m[
    "all_fits_patience_valid"
] is True


common = json.loads(
    COMMON_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert common[
    "status"
] == "COMMON_COHORT_AND_SCALERS_COMPLETE_BEFORE_MODELING"

assert common[
    "same_row_ids_all_conditions"
] is True

assert common[
    "scaler_contract"
][
    "validation_rows_used"
] is False

assert common[
    "scaler_contract"
][
    "test_rows_used"
] is False

assert common[
    "models_trained"
] is False


# ------------------------------------------------------------
# Parameter-count contract.
#
# Dense stack:
# input -> 64 -> 64 -> 1
# causal aggregation is parameter-free.
# ------------------------------------------------------------

def parameter_count(
    input_dim,
    hidden=64,
):
    return (
        input_dim
        * hidden
        + hidden
        + hidden
        * hidden
        + hidden
        + hidden
        + 1
    )


assert parameter_count(
    20
) == 5569

assert parameter_count(
    30
) == 6209


conditions = {
    "PRIMARY20_COMMON": {
        "input_dimension":
            20,

        "parameter_count_each_model":
            5569,

        "feature_blocks": [
            "PRIMARY20",
        ],

        "interpretation":
            "feature-matched primary reference on the common cohort",
    },

    "PRIMARY20_PLUS_FULL_DQDV10": {
        "input_dimension":
            30,

        "parameter_count_each_model":
            6209,

        "feature_blocks": [
            "PRIMARY20",
            "FULL_DQDV10",
        ],

        "interpretation":
            (
                "retrospective leakage-positive control; "
                "not deployment-valid"
            ),
    },

    "PRIMARY20_PLUS_PREFIX25_DQDV10": {
        "input_dimension":
            30,

        "parameter_count_each_model":
            6209,

        "feature_blocks": [
            "PRIMARY20",
            "PREFIX25_DQDV10",
        ],

        "interpretation":
            (
                "partial-curve control using split-specific "
                "training-only fixed time cutoff"
            ),
    },

    "PRIMARY20_PLUS_PREFIX50_DQDV10": {
        "input_dimension":
            30,

        "parameter_count_each_model":
            6209,

        "feature_blocks": [
            "PRIMARY20",
            "PREFIX50_DQDV10",
        ],

        "interpretation":
            (
                "partial-curve control using split-specific "
                "training-only fixed time cutoff"
            ),
    },
}


planned = []

for condition in conditions:

    for split_seed in [
        0, 1, 2, 3, 4
    ]:

        for train_seed in [
            0, 1, 2
        ]:

            planned.append({
                "condition":
                    condition,

                "split_seed":
                    split_seed,

                "train_seed":
                    train_seed,
            })


assert len(
    planned
) == 60

expected_model_fits = (
    len(
        planned
    )
    * 2
)

assert expected_model_fits == 120


protocol = {
    "contract":
        "dqdv_training_protocol_v1",

    "status":
        "FROZEN_BEFORE_MODEL_TRAINING",

    "scientific_question":
        (
            "Does full-discharge dQ/dV create target leakage, and "
            "how much of any apparent gain remains when dQ/dV is "
            "restricted to fixed early discharge windows whose "
            "durations are fitted using training rows only?"
        ),

    "upstream_anchors": {
        "dqdv_leakage_control_protocol_sha256":
            EXPECTED[
                "dqdv_protocol"
            ],

        "feature_bank_manifest_sha256":
            EXPECTED[
                "feature_bank_manifest"
            ],

        "common_cohort_scaler_manifest_sha256":
            EXPECTED[
                "common_manifest"
            ],

        "final_v3_training_protocol_sha256":
            EXPECTED[
                "final_v3_protocol"
            ],

        "final_v3_execution_manifest_sha256":
            EXPECTED[
                "final_v3_manifest"
            ],

        "final_v3_runner_sha256":
            EXPECTED[
                "final_v3_runner"
            ],

        "feature_matched_model_implementation_sha256":
            sha256(
                MODEL_IMPL
            ),
    },

    "data_contract": {
        "task":
            "cycle-level SOH",

        "datasets": [
            "HUST",
            "CALCE",
            "MATR",
        ],

        "canonical_soh_eligible_rows_before_common_cohort":
            300553,

        "common_cohort":
            (
                "split-specific intersection of PRIMARY20, FULL, "
                "PREFIX25, and PREFIX50 valid rows"
            ),

        "same_row_ids_all_four_conditions":
            True,

        "primary_scaler":
            (
                "one training-only PRIMARY20 scaler per split seed "
                "reused across all four conditions"
            ),

        "dqdv_scaler":
            (
                "separate training-only dQ/dV10 scaler for FULL, "
                "PREFIX25, and PREFIX50 within each split seed"
            ),

        "validation_rows_used_for_scaler_fit":
            False,

        "test_rows_used_for_scaler_fit":
            False,
    },

    "conditions":
        conditions,

    "model_contract": {
        "models": [
            "FeatureMatchedMLP",
            "FeatureMatchedCausalGCN",
        ],

        "architecture":
            "input_dim -> 64 -> 64 -> 1",

        "activation":
            "ReLU",

        "batch_normalization":
            False,

        "dropout":
            False,

        "causal_gcn_operator":
            (
                "parameter-free causal mean aggregation over "
                "{self, previous valid observation} before fc1 "
                "and fc2"
            ),

        "causal_receptive_field":
            "at most {t,t-1,t-2}",

        "future_edges":
            False,

        "cross_cell_edges":
            False,

        "parameter_matching":
            (
                "MLP and causal GCN have identical learnable "
                "parameter counts within each condition"
            ),

        "same_train_seed_initialization_within_pair":
            True,
    },

    "training_contract": {
        "device":
            "cpu",

        "optimizer":
            "AdamW",

        "learning_rate":
            0.001,

        "weight_decay":
            1e-5,

        "objective":
            "dataset-balanced + cell-balanced MSE",

        "early_stopping_metric":
            "validation dataset-balanced cell-balanced RMSE",

        "patience":
            30,

        "max_epochs":
            5000,

        "max_epochs_role":
            (
                "nonbinding safety ceiling inherited from final-v3"
            ),

        "new_hyperparameter_tuning":
            False,

        "split_seeds": [
            0, 1, 2, 3, 4
        ],

        "training_initialization_seeds": [
            0, 1, 2
        ],

        "planned_condition_split_init_combinations":
            60,

        "models_per_combination":
            2,

        "expected_model_fits":
            120,
    },

    "evaluation_contract": {
        "model_selection_uses_test_data":
            False,

        "test_evaluation_allowed_only_after_model_selection":
            True,

        "test_metric_values_may_be_written_to_artifacts":
            True,

        "test_metric_values_may_be_used_to_change_protocol":
            False,

        "test_metric_values_may_be_surfaced_before_execution_audit":
            False,

        "balanced_all_metric_semantics":
            (
                "equal-dataset, equal-cell, equal-observation "
                "hierarchical weighted_metrics"
            ),

        "dataset_specific_metric_semantics":
            "plain row-level unweighted dataset_metrics",

        "metrics": [
            "RMSE",
            "MAE",
        ],
    },

    "interpretation_contract": {
        "full_dqdv":
            (
                "positive leakage control because full-discharge "
                "ica_area_Ah can encode discharge capacity"
            ),

        "prefix25_and_prefix50":
            (
                "target-independent partial-curve controls generated "
                "from fixed training-derived elapsed-time windows"
            ),

        "feature_dimension_difference_disclosed":
            True,

        "primary20_dimension":
            20,

        "dqdv_augmented_dimension":
            30,

        "inferential_tests_added":
            False,

        "analysis_role":
            "descriptive leakage-control comparison",

        "universal_gcn_superiority_claim_allowed":
            False,
    },

    "planned_runs":
        planned,

    "pretraining_barriers": {
        "target_values_read_during_protocol_freeze":
            False,

        "test_metric_values_read_during_protocol_freeze":
            False,

        "model_training_performed_during_protocol_freeze":
            False,
    },

    "freezer": {
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
}


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_json(
    protocol,
    PROTOCOL_OUT,
)


print(
    "DQDV TRAINING PROTOCOL FREEZE"
)

print(
    "conditions:",
    len(
        conditions
    )
)

print(
    "planned_condition_split_init_combinations:",
    len(
        planned
    )
)

print(
    "expected_model_fits:",
    expected_model_fits
)

print(
    "primary20_parameter_count_each_model:",
    5569
)

print(
    "augmented30_parameter_count_each_model:",
    6209
)

print(
    "optimizer:",
    "AdamW"
)

print(
    "learning_rate:",
    0.001
)

print(
    "weight_decay:",
    1e-5
)

print(
    "max_epochs:",
    5000
)

print(
    "patience:",
    30
)

print(
    "new_hyperparameter_tuning:",
    False
)

print(
    "same_row_ids_all_four_conditions:",
    True
)

print(
    "test_metric_values_read:",
    False
)

print(
    "model_training_performed:",
    False
)

print(
    "protocol_sha256:",
    sha256(
        PROTOCOL_OUT
    )
)

print(
    "DQDV_TRAINING_PROTOCOL_FREEZE_PASS=True"
)
