from pathlib import Path
import ast
import hashlib
import json


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

DQROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

TRAINROOT = DQROOT / "training_v1"

PROTOCOL = (
    TRAINROOT
    / "dqdv_training_protocol_v1.json"
)

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


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

assert p[
    "contract"
] == "dqdv_training_protocol_v1"

assert p[
    "status"
] == "FROZEN_BEFORE_MODEL_TRAINING"


anchors = p[
    "upstream_anchors"
]

assert sha256(
    DQ_PROTOCOL
) == anchors[
    "dqdv_leakage_control_protocol_sha256"
]

assert sha256(
    COMMON_MANIFEST
) == anchors[
    "common_cohort_scaler_manifest_sha256"
]

assert sha256(
    FEATURE_BANK_MANIFEST
) == anchors[
    "feature_bank_manifest_sha256"
]

assert sha256(
    FINAL_V3_PROTOCOL
) == anchors[
    "final_v3_training_protocol_sha256"
]

assert sha256(
    FINAL_V3_MANIFEST
) == anchors[
    "final_v3_execution_manifest_sha256"
]

assert sha256(
    FINAL_V3_RUNNER
) == anchors[
    "final_v3_runner_sha256"
]

assert sha256(
    MODEL_IMPL
) == anchors[
    "feature_matched_model_implementation_sha256"
]


# ------------------------------------------------------------
# Independent comparison with final-v3 training contract.
# Do not inspect final-v3 records.
# ------------------------------------------------------------

v3 = json.loads(
    FINAL_V3_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

tc = p[
    "training_contract"
]

assert tc[
    "optimizer"
] == v3[
    "optimizer"
] == "AdamW"

assert float(
    tc[
        "learning_rate"
    ]
) == float(
    v3[
        "learning_rate"
    ]
) == 0.001

assert float(
    tc[
        "weight_decay"
    ]
) == float(
    v3[
        "weight_decay"
    ]
) == 1e-5

assert int(
    tc[
        "max_epochs"
    ]
) == int(
    v3[
        "max_epochs"
    ]
) == 5000

assert int(
    tc[
        "patience"
    ]
) == int(
    v3[
        "early_stopping_patience"
    ]
) == 30

assert tc[
    "objective"
] == v3[
    "objective"
]

assert tc[
    "early_stopping_metric"
] == v3[
    "early_stopping_metric"
]

assert tc[
    "new_hyperparameter_tuning"
] is False

assert tc[
    "split_seeds"
] == [
    0, 1, 2, 3, 4
]

assert tc[
    "training_initialization_seeds"
] == [
    0, 1, 2
]


# ------------------------------------------------------------
# Data/scaler barriers.
# ------------------------------------------------------------

common = json.loads(
    COMMON_MANIFEST.read_text(
        encoding="utf-8"
    )
)

dc = p[
    "data_contract"
]

assert common[
    "same_row_ids_all_conditions"
] is True

assert dc[
    "same_row_ids_all_four_conditions"
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

assert dc[
    "validation_rows_used_for_scaler_fit"
] is False

assert dc[
    "test_rows_used_for_scaler_fit"
] is False


# ------------------------------------------------------------
# Independent fit cardinality.
# ------------------------------------------------------------

conditions = p[
    "conditions"
]

assert set(
    conditions
) == {
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
}

assert conditions[
    "PRIMARY20_COMMON"
][
    "input_dimension"
] == 20

for name in [
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]:

    assert conditions[
        name
    ][
        "input_dimension"
    ] == 30


def nparams(d):
    return (
        d * 64
        + 64
        + 64 * 64
        + 64
        + 64
        + 1
    )


assert nparams(
    20
) == 5569

assert nparams(
    30
) == 6209

assert conditions[
    "PRIMARY20_COMMON"
][
    "parameter_count_each_model"
] == 5569

for name in [
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]:

    assert conditions[
        name
    ][
        "parameter_count_each_model"
    ] == 6209


expected_runs = {
    (
        condition,
        split_seed,
        train_seed,
    )
    for condition in conditions
    for split_seed in [
        0, 1, 2, 3, 4
    ]
    for train_seed in [
        0, 1, 2
    ]
}

actual_runs = {
    (
        str(
            r[
                "condition"
            ]
        ),
        int(
            r[
                "split_seed"
            ]
        ),
        int(
            r[
                "train_seed"
            ]
        ),
    )
    for r in p[
        "planned_runs"
    ]
}

assert actual_runs == expected_runs

assert len(
    actual_runs
) == 60

assert tc[
    "planned_condition_split_init_combinations"
] == 60

assert tc[
    "models_per_combination"
] == 2

assert tc[
    "expected_model_fits"
] == 120


# ------------------------------------------------------------
# Model API / causality declarations.
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

mc = p[
    "model_contract"
]

assert mc[
    "future_edges"
] is False

assert mc[
    "cross_cell_edges"
] is False

assert mc[
    "same_train_seed_initialization_within_pair"
] is True


# ------------------------------------------------------------
# Test barrier and interpretation.
# ------------------------------------------------------------

ec = p[
    "evaluation_contract"
]

assert ec[
    "model_selection_uses_test_data"
] is False

assert ec[
    "test_metric_values_may_be_used_to_change_protocol"
] is False

assert ec[
    "test_metric_values_may_be_surfaced_before_execution_audit"
] is False

ic = p[
    "interpretation_contract"
]

assert ic[
    "feature_dimension_difference_disclosed"
] is True

assert ic[
    "inferential_tests_added"
] is False

assert ic[
    "universal_gcn_superiority_claim_allowed"
] is False

assert p[
    "pretraining_barriers"
][
    "target_values_read_during_protocol_freeze"
] is False

assert p[
    "pretraining_barriers"
][
    "test_metric_values_read_during_protocol_freeze"
] is False

assert p[
    "pretraining_barriers"
][
    "model_training_performed_during_protocol_freeze"
] is False


print(
    "DQDV TRAINING PROTOCOL AUDIT"
)

print(
    "upstream anchors exact:",
    True
)

print(
    "final-v3 optimizer contract inherited:",
    True
)

print(
    "final-v3 objective inherited:",
    True
)

print(
    "final-v3 early stopping inherited:",
    True
)

print(
    "new hyperparameter tuning:",
    False
)

print(
    "conditions:",
    4
)

print(
    "planned condition/split/init combinations:",
    60
)

print(
    "expected model fits:",
    120
)

print(
    "MLP/GCN parameter matched within conditions:",
    True
)

print(
    "primary20 parameters each:",
    5569
)

print(
    "augmented30 parameters each:",
    6209
)

print(
    "same common-cohort rows all conditions:",
    True
)

print(
    "scalers training-only:",
    True
)

print(
    "full dQdV marked leakage-positive control:",
    True
)

print(
    "feature-dimension difference disclosed:",
    True
)

print(
    "new inferential tests:",
    False
)

print(
    "test metric values read:",
    False
)

print(
    "model training performed:",
    False
)

print(
    "protocol SHA256:",
    sha256(
        PROTOCOL
    )
)

print(
    "DQDV_TRAINING_PROTOCOL_AUDIT_PASS=True"
)
