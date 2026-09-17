from pathlib import Path
import ast
import hashlib
import json


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

V1 = (
    ROOT
    / "training_v1/dqdv_training_protocol_v1.json"
)

V2 = (
    ROOT
    / "training_v2/dqdv_training_protocol_v2.json"
)

S24 = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

EXPECTED_V1_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
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

            h.update(
                b
            )

    return h.hexdigest()


assert sha256(
    V1
) == EXPECTED_V1_SHA


a = json.loads(
    V1.read_text(
        encoding="utf-8"
    )
)

b = json.loads(
    V2.read_text(
        encoding="utf-8"
    )
)


assert b[
    "contract"
] == "dqdv_training_protocol_v2"

assert b[
    "parent_protocol_sha256"
] == EXPECTED_V1_SHA

assert b[
    "status"
] == (
    "FROZEN_AFTER_VALIDATION_ONLY_CEILING_AMENDMENT_BEFORE_RESUME"
)


# Only intended training-control change.
assert int(
    a[
        "training_contract"
    ][
        "max_epochs"
    ]
) == 5000

assert int(
    b[
        "training_contract"
    ][
        "max_epochs"
    ]
) == 10000

for key in [
    "device",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "objective",
    "early_stopping_metric",
    "patience",
    "new_hyperparameter_tuning",
    "split_seeds",
    "training_initialization_seeds",
    "planned_condition_split_init_combinations",
    "models_per_combination",
    "expected_model_fits",
]:

    assert (
        a[
            "training_contract"
        ][
            key
        ]
        ==
        b[
            "training_contract"
        ][
            key
        ]
    )


assert b[
    "training_contract"
][
    "patience"
] == 30

assert b[
    "training_contract"
][
    "new_hyperparameter_tuning"
] is False


# Data/model/evaluation contracts unchanged.
for key in [
    "data_contract",
    "conditions",
    "model_contract",
    "evaluation_contract",
    "interpretation_contract",
    "planned_runs",
]:

    assert a[
        key
    ] == b[
        key
    ]


c = b[
    "ceiling_amendment"
]

assert c[
    "old_max_epochs"
] == 5000

assert c[
    "new_max_epochs"
] == 10000

assert c[
    "patience_unchanged"
] == 30

assert c[
    "test_metric_values_used_for_amendment"
] is False

assert c[
    "test_metric_values_surfaced_for_amendment"
] is False

assert c[
    "test_metric_values_inspected_for_amendment"
] is False

assert c[
    "completed_v1_pairs_compatible_with_v2"
] is True

assert c[
    "compatible_completed_pair_count"
] == 4

assert c[
    "compatible_completed_model_fit_count"
] == 8

assert c[
    "remaining_pairs_to_execute_under_v2"
] == 56

assert c[
    "remaining_model_fits_to_execute_under_v2"
] == 112

assert c[
    "invalid_partial_v1_states_reused"
] is False

assert len(
    c[
        "invalid_partial_v1_states"
    ]
) == 2

assert c[
    "failed_pair"
][
    "must_rerun_from_scratch_under_v2"
] is True


# All compatible records still exist with exact hashes.
assert len(
    c[
        "compatible_v1_records"
    ]
) == 4

for r in c[
    "compatible_v1_records"
]:

    rp = Path(
        r[
            "record_path"
        ]
    )

    assert rp.is_file()

    assert sha256(
        rp
    ) == r[
        "record_sha256"
    ]

    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        m = r[
            "models"
        ][
            model_name
        ]

        assert int(
            m[
                "stop_epoch"
            ]
        ) < 5000

        assert (
            int(
                m[
                    "stop_epoch"
                ]
            )
            -
            int(
                m[
                    "best_epoch"
                ]
            )
        ) == 30

        sp = Path(
            m[
                "state_path"
            ]
        )

        assert sp.is_file()

        assert sha256(
            sp
        ) == m[
            "state_sha256"
        ]


# Incomplete states preserved but explicitly non-reusable.
for item in c[
    "invalid_partial_v1_states"
]:

    assert item[
        "reusable_in_v2"
    ] is False

    p = Path(
        item[
            "path"
        ]
    )

    assert p.is_file()

    assert sha256(
        p
    ) == item[
        "sha256"
    ]


# Independently verify static termination logic.
src = S24.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    src
)

train = next(
    n
    for n in tree.body
    if isinstance(
        n,
        ast.FunctionDef,
    )
    and n.name == "train_one"
)

breaks = [
    n
    for n in ast.walk(
        train
    )
    if isinstance(
        n,
        ast.Break,
    )
]

assert len(
    breaks
) == 1

train_source = ast.get_source_segment(
    src,
    train,
)

assert "if wait >= patience:" in train_source

assert "max_epochs + 1" in train_source


assert b[
    "pretraining_barriers_v2"
][
    "model_training_performed_during_v2_freeze"
] is False

assert b[
    "pretraining_barriers_v2"
][
    "test_metric_values_read_during_v2_freeze"
] is False

assert b[
    "pretraining_barriers_v2"
][
    "amendment_uses_validation_termination_only"
] is True


print(
    "DQDV TRAINING PROTOCOL V2 AUDIT"
)

print(
    "parent protocol exact:",
    True
)

print(
    "only intended training-control change is ceiling:",
    True
)

print(
    "max_epochs 5000 -> 10000:",
    True
)

print(
    "patience unchanged at 30:",
    True
)

print(
    "optimizer/lr/weight decay/objective unchanged:",
    True
)

print(
    "data/model/evaluation contracts unchanged:",
    True
)

print(
    "validation-only amendment basis:",
    True
)

print(
    "test metrics inspected for amendment:",
    False
)

print(
    "compatible completed v1 pairs independently verified:",
    4
)

print(
    "compatible completed fits independently verified:",
    8
)

print(
    "incomplete pair must rerun:",
    True
)

print(
    "incomplete states preserved but not reusable:",
    True
)

print(
    "protocol_v2_sha256:",
    sha256(
        V2
    )
)

print(
    "DQDV_TRAINING_PROTOCOL_V2_AUDIT_PASS=True"
)
