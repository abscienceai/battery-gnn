from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import subprocess


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

V1_PROTOCOL = (
    ROOT
    / "training_v1/dqdv_training_protocol_v1.json"
)

V1_EXEC = (
    ROOT
    / "training_v1/final_execution_v1"
)

STATUS = (
    V1_EXEC
    / "execution_status.json"
)

FAIL_LOG = (
    BASE
    / "logs/97_final_dqdv_training_execution_resume_3of60.txt"
)

S24 = (
    REPO
    / "scripts/revision_2026/24_feature_matched_soh_benchmark.py"
)

RUNNER = (
    REPO
    / "scripts/revision_2026/89_final_dqdv_training_runner.py"
)

SELF = (
    REPO
    / "scripts/revision_2026/91_freeze_dqdv_training_protocol_v2.py"
)

OUT = (
    ROOT
    / "training_v2"
)

V2_PROTOCOL = (
    OUT
    / "dqdv_training_protocol_v2.json"
)

EXPECTED_V1_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
)

EXPECTED_RUNNER_SHA = (
    "117f5dea393cb27c487e07e7a0ffc7c46195bb71d69c7809fcc574517caa573f"
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


assert sha256(
    V1_PROTOCOL
) == EXPECTED_V1_SHA

assert sha256(
    RUNNER
) == EXPECTED_RUNNER_SHA


v1 = json.loads(
    V1_PROTOCOL.read_text(
        encoding="utf-8"
    )
)

status = json.loads(
    STATUS.read_text(
        encoding="utf-8"
    )
)


# ------------------------------------------------------------
# Failure-state evidence.
# ------------------------------------------------------------

assert int(
    status[
        "completed_pairs"
    ]
) == 4

assert int(
    status[
        "completed_model_fits"
    ]
) == 8

assert status[
    "current"
] == {
    "condition":
        "PRIMARY20_PLUS_FULL_DQDV10",

    "split_seed":
        0,

    "train_seed":
        1,
}

assert status[
    "test_metric_values_surfaced"
] is False


# ------------------------------------------------------------
# Static termination proof:
# train_one has exactly one explicit break, and it is controlled by
# wait >= patience. Otherwise the epoch loop exhausts max_epochs.
# ------------------------------------------------------------

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

assert train_source is not None

assert "if wait >= patience:" in train_source

assert "for epoch in range(" in train_source

assert "max_epochs + 1" in train_source


# ------------------------------------------------------------
# Existing valid pair records.
# ------------------------------------------------------------

record_files = sorted(
    (
        V1_EXEC
        / "run_records"
    ).glob(
        "*.json"
    )
)

assert len(
    record_files
) == 4


expected_pairs = {
    (
        "PRIMARY20_COMMON",
        0,
        0,
    ),
    (
        "PRIMARY20_COMMON",
        0,
        1,
    ),
    (
        "PRIMARY20_COMMON",
        0,
        2,
    ),
    (
        "PRIMARY20_PLUS_FULL_DQDV10",
        0,
        0,
    ),
}


seen_pairs = set()

compatible_records = []

used_state_paths = set()


for path in record_files:

    r = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    key = (
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

    seen_pairs.add(
        key
    )

    assert key in expected_pairs

    assert r[
        "protocol_sha256"
    ] == EXPECTED_V1_SHA

    assert r[
        "test_metric_values_used_to_change_protocol"
    ] is False

    assert r[
        "test_metric_values_surfaced_during_execution"
    ] is False


    model_summary = {}

    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        m = r[
            "models"
        ][
            model_name
        ]

        assert m[
            "patience_valid"
        ] is True

        assert m[
            "below_epoch_ceiling"
        ] is True

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

        used_state_paths.add(
            sp.resolve()
        )

        model_summary[
            model_name
        ] = {
            "best_epoch":
                int(
                    m[
                        "best_epoch"
                    ]
                ),

            "stop_epoch":
                int(
                    m[
                        "stop_epoch"
                    ]
                ),

            "state_path":
                str(
                    sp
                ),

            "state_sha256":
                m[
                    "state_sha256"
                ],
        }


    compatible_records.append({
        "condition":
            key[
                0
            ],

        "split_seed":
            key[
                1
            ],

        "train_seed":
            key[
                2
            ],

        "record_path":
            str(
                path
            ),

        "record_sha256":
            sha256(
                path
            ),

        "models":
            model_summary,
    })


assert seen_pairs == expected_pairs


# ------------------------------------------------------------
# Two states from the incomplete pair exist but are NOT reusable.
# ------------------------------------------------------------

all_state_files = sorted(
    (
        V1_EXEC
        / "model_states"
    ).glob(
        "*.pt"
    )
)

assert len(
    all_state_files
) == 10


extra_states = [
    p
    for p in all_state_files
    if p.resolve()
    not in used_state_paths
]

assert len(
    extra_states
) == 2


expected_extra_names = {
    "PRIMARY20_PLUS_FULL_DQDV10__split0__train1__MLP.pt",
    "PRIMARY20_PLUS_FULL_DQDV10__split0__train1__CAUSAL_GCN.pt",
}

assert {
    p.name
    for p in extra_states
} == expected_extra_names


invalid_partial_states = [
    {
        "path":
            str(
                p
            ),

        "sha256":
            sha256(
                p
            ),

        "reusable_in_v2":
            False,
    }
    for p in extra_states
]


# ------------------------------------------------------------
# Failure log confirms pair did not close.
# No numerical test metric values are parsed.
# ------------------------------------------------------------

log_text = FAIL_LOG.read_text(
    encoding="utf-8"
)

assert (
    "PRIMARY20_PLUS_FULL_DQDV10 "
    "split_seed=0 train_seed=0 completed=4/60"
) in log_text

assert not (
    "PRIMARY20_PLUS_FULL_DQDV10 "
    "split_seed=0 train_seed=1 completed=5/60"
) in log_text

assert "AssertionError" in log_text


# ------------------------------------------------------------
# Freeze protocol v2.
# Only max_epochs safety ceiling changes.
# ------------------------------------------------------------

v2 = copy.deepcopy(
    v1
)

v2[
    "contract"
] = "dqdv_training_protocol_v2"

v2[
    "status"
] = (
    "FROZEN_AFTER_VALIDATION_ONLY_CEILING_AMENDMENT_BEFORE_RESUME"
)

v2[
    "parent_protocol_sha256"
] = EXPECTED_V1_SHA


v2[
    "training_contract"
][
    "max_epochs"
] = 10000

v2[
    "training_contract"
][
    "max_epochs_role"
] = (
    "nonbinding safety ceiling; extended from 5000 to 10000 "
    "after validation-only termination evidence showed that the "
    "old 5000-epoch ceiling was binding for one 30D MLP fit; "
    "30-epoch validation patience remains the intended stopping rule"
)

v2[
    "training_contract"
][
    "new_hyperparameter_tuning"
] = False


v2[
    "ceiling_amendment"
] = {
    "old_max_epochs":
        5000,

    "new_max_epochs":
        10000,

    "patience_unchanged":
        30,

    "learning_rate_unchanged":
        0.001,

    "weight_decay_unchanged":
        1e-5,

    "optimizer_unchanged":
        "AdamW",

    "objective_unchanged":
        "dataset-balanced + cell-balanced MSE",

    "early_stopping_metric_unchanged":
        "validation dataset-balanced cell-balanced RMSE",

    "amendment_basis":
        (
            "validation-only termination evidence; MLP patience_valid "
            "assertion failed after train_one returned, while train_one "
            "has exactly one explicit early break at wait >= patience; "
            "therefore the old 5000-epoch safety ceiling was exhausted"
        ),

    "test_metric_values_used_for_amendment":
        False,

    "test_metric_values_surfaced_for_amendment":
        False,

    "test_metric_values_may_have_been_computed_in_memory_for_incomplete_pair":
        True,

    "test_metric_values_inspected_for_amendment":
        False,

    "failed_pair":
        {
            "condition":
                "PRIMARY20_PLUS_FULL_DQDV10",

            "split_seed":
                0,

            "train_seed":
                1,

            "must_rerun_from_scratch_under_v2":
                True,
        },

    "completed_v1_pairs_compatible_with_v2":
        True,

    "compatibility_reason":
        (
            "all four completed v1 pairs terminated by the unchanged "
            "30-epoch patience rule before epoch 5000; increasing only "
            "the safety ceiling cannot alter those deterministic "
            "already-terminated trajectories"
        ),

    "compatible_completed_pair_count":
        4,

    "compatible_completed_model_fit_count":
        8,

    "remaining_pairs_to_execute_under_v2":
        56,

    "remaining_model_fits_to_execute_under_v2":
        112,

    "compatible_v1_records":
        compatible_records,

    "invalid_partial_v1_states":
        invalid_partial_states,

    "invalid_partial_v1_states_preserved_for_provenance":
        True,

    "invalid_partial_v1_states_reused":
        False,

    "source_execution_status_sha256":
        sha256(
            STATUS
        ),

    "source_failure_log_sha256":
        sha256(
            FAIL_LOG
        ),

    "source_runner_sha256":
        EXPECTED_RUNNER_SHA,

    "source_train_one_sha256":
        sha256(
            S24
        ),
}


v2[
    "pretraining_barriers_v2"
] = {
    "model_training_performed_during_v2_freeze":
        False,

    "test_metric_values_read_during_v2_freeze":
        False,

    "test_metric_values_surfaced_during_v2_freeze":
        False,

    "amendment_uses_validation_termination_only":
        True,
}


v2[
    "v2_freezer"
] = {
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
}


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_json(
    v2,
    V2_PROTOCOL,
)


print(
    "DQDV TRAINING PROTOCOL V2 FREEZE"
)

print(
    "parent protocol exact:",
    True
)

print(
    "old max_epochs:",
    5000
)

print(
    "new max_epochs:",
    10000
)

print(
    "patience unchanged:",
    30
)

print(
    "optimizer/objective unchanged:",
    True
)

print(
    "amendment basis validation-only:",
    True
)

print(
    "test metrics inspected for amendment:",
    False
)

print(
    "compatible completed v1 pairs:",
    4
)

print(
    "compatible completed v1 model fits:",
    8
)

print(
    "incomplete v1 state files preserved:",
    2
)

print(
    "incomplete v1 states reusable:",
    False
)

print(
    "remaining pairs under v2:",
    56
)

print(
    "model training during freeze:",
    False
)

print(
    "protocol_v2_sha256:",
    sha256(
        V2_PROTOCOL
    )
)

print(
    "DQDV_TRAINING_PROTOCOL_V2_FREEZE_PASS=True"
)
