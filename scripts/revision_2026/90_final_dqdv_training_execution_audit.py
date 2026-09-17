from pathlib import Path
import hashlib
import json
import math

import numpy as np


BASE = Path.home() / "alper/battery_revision_2026"
REPO = BASE / "repo/battery-gnn"

ROOT = (
    BASE
    / "results/revision_2026/dqdv_leakage_control/final_dqdv_v1"
)

TRAINROOT = ROOT / "training_v1"
EXEC = TRAINROOT / "final_execution_v1"

PROTOCOL = TRAINROOT / "dqdv_training_protocol_v1.json"

COMMON_MANIFEST = (
    ROOT
    / "common_cohort_scalers_v1/common_cohort_scaler_manifest.json"
)

MANIFEST = EXEC / "execution_manifest.json"

RUNNER = (
    REPO
    / "scripts/revision_2026/89_final_dqdv_training_runner.py"
)

EXPECTED_PROTOCOL_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
)

EXPECTED_COMMON_MANIFEST_SHA = (
    "3ee48af31ad580c14de0460fa1f163ad572e369368d6da71c0ba6823d28905a1"
)

COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA = (
    "46f1ad8ab073e320b5ca5420283cec15c5b2d661f3f2089b41716a5b4d72a76d"
)

CONDITIONS = [
    "PRIMARY20_COMMON",
    "PRIMARY20_PLUS_FULL_DQDV10",
    "PRIMARY20_PLUS_PREFIX25_DQDV10",
    "PRIMARY20_PLUS_PREFIX50_DQDV10",
]

DATASETS = {
    "HUST",
    "CALCE",
    "MATR",
}

SPLITS = {
    "train",
    "validation",
    "test",
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


def assert_finite_tree(x):

    if isinstance(
        x,
        dict,
    ):

        for v in x.values():
            assert_finite_tree(
                v
            )

        return

    if isinstance(
        x,
        list,
    ):

        for v in x:
            assert_finite_tree(
                v
            )

        return

    if isinstance(
        x,
        bool,
    ):
        return

    if isinstance(
        x,
        (
            int,
            float,
        ),
    ):

        assert math.isfinite(
            float(
                x
            )
        )

        return

    if isinstance(
        x,
        (
            str,
            type(None),
        ),
    ):
        return

    raise AssertionError(
        type(
            x
        )
    )


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    COMMON_MANIFEST
) == EXPECTED_COMMON_MANIFEST_SHA


p = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert m[
    "contract"
] == "dqdv_training_execution_v1"

assert m[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "common_cohort_manifest_sha256"
] == EXPECTED_COMMON_MANIFEST_SHA

assert sha256(
    RUNNER
) == m[
    "runner_sha256"
]

assert m[
    "planned_pair_count"
] == 60

assert m[
    "completed_pair_count"
] == 60

assert m[
    "planned_model_fit_count"
] == 120

assert m[
    "completed_model_fit_count"
] == 120

assert m[
    "epoch_ceiling"
] == 5000

assert m[
    "patience"
] == 30

assert m[
    "all_fits_patience_valid"
] is True

assert m[
    "all_fits_below_epoch_ceiling"
] is True

assert m[
    "test_metric_values_parsed_during_closure"
] is False

assert m[
    "test_metric_values_surfaced_during_execution"
] is False

assert m[
    "test_metric_values_used_to_change_protocol"
] is False

assert m[
    "protocol_changed_after_test_evaluation"
] is False


expected_pairs = {
    (
        condition,
        split_seed,
        train_seed,
    )
    for condition in CONDITIONS
    for split_seed in range(
        5
    )
    for train_seed in range(
        3
    )
}

manifest_pairs = {
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
    for r in m[
        "records"
    ]
}

assert manifest_pairs == expected_pairs

assert len(
    m[
        "records"
    ]
) == 60


data_counts_by_seed = {}

record_count = 0
fit_count = 0
compatible_prefixed_runner_records = 0


for summary in m[
    "records"
]:

    condition = str(
        summary[
            "condition"
        ]
    )

    split_seed = int(
        summary[
            "split_seed"
        ]
    )

    train_seed = int(
        summary[
            "train_seed"
        ]
    )

    rp = Path(
        summary[
            "record_path"
        ]
    )

    assert rp.is_file()

    assert sha256(
        rp
    ) == summary[
        "record_sha256"
    ]


    r = json.loads(
        rp.read_text(
            encoding="utf-8"
        )
    )


    assert r[
        "contract"
    ] == "dqdv_training_pair_record_v1"

    assert r[
        "protocol_sha256"
    ] == EXPECTED_PROTOCOL_SHA

    assert r[
        "condition"
    ] == condition

    record_runner_sha = str(
        r[
            "runner_sha256"
        ]
    )

    if (
        record_runner_sha
        != m[
            "runner_sha256"
        ]
    ):

        assert (
            record_runner_sha
            == COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA
        )

        assert condition == "PRIMARY20_COMMON"
        assert split_seed == 0
        assert train_seed in {
            0,
            1,
            2,
        }

        assert int(
            r[
                "input_dimension"
            ]
        ) == 20

        compatible_prefixed_runner_records += 1

    assert int(
        r[
            "split_seed"
        ]
    ) == split_seed

    assert int(
        r[
            "train_seed"
        ]
    ) == train_seed

    assert r[
        "initial_weights_equal_within_pair"
    ] is True

    assert r[
        "model_selection_uses_test_data"
    ] is False

    assert r[
        "test_metric_values_used_to_change_protocol"
    ] is False

    assert r[
        "test_metric_values_surfaced_during_execution"
    ] is False


    counts = {
        str(k):
            int(v)
        for k, v in r[
            "data_counts"
        ].items()
    }

    assert set(
        counts
    ) == SPLITS

    assert all(
        v > 0
        for v in counts.values()
    )


    if split_seed not in data_counts_by_seed:

        data_counts_by_seed[
            split_seed
        ] = counts

    else:

        assert (
            data_counts_by_seed[
                split_seed
            ]
            == counts
        )


    expected_dim = (
        20
        if condition == "PRIMARY20_COMMON"
        else 30
    )

    expected_params = (
        5569
        if expected_dim == 20
        else 6209
    )

    assert int(
        r[
            "input_dimension"
        ]
    ) == expected_dim

    assert set(
        r[
            "models"
        ]
    ) == {
        "MLP",
        "CAUSAL_GCN",
    }


    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        q = r[
            "models"
        ][
            model_name
        ]

        fit_count += 1


        assert int(
            q[
                "parameter_count"
            ]
        ) == expected_params

        best_epoch = int(
            q[
                "best_epoch"
            ]
        )

        stop_epoch = int(
            q[
                "stop_epoch"
            ]
        )

        assert stop_epoch < 5000

        assert (
            stop_epoch
            - best_epoch
        ) == 30

        assert int(
            q[
                "epochs_after_best"
            ]
        ) == 30

        assert q[
            "patience_valid"
        ] is True

        assert q[
            "below_epoch_ceiling"
        ] is True


        history = q[
            "history"
        ]

        assert len(
            history
        ) > 0

        assert int(
            history[
                -1
            ][
                "epoch"
            ]
        ) == stop_epoch

        assert_finite_tree(
            history
        )


        sp = Path(
            q[
                "state_path"
            ]
        )

        assert sp.is_file()

        assert sha256(
            sp
        ) == q[
            "state_sha256"
        ]


        metrics = q[
            "metrics"
        ]

        assert set(
            metrics
        ) == SPLITS


        # Test metrics are parsed here for the first independent
        # execution audit, but numerical values are intentionally
        # not printed before this audit passes.
        for split_name in SPLITS:

            sm = metrics[
                split_name
            ]

            assert set(
                sm
            ) == {
                "balanced",
                "by_dataset",
            }


            balanced = sm[
                "balanced"
            ]

            assert {
                "rmse",
                "mae",
                "mse",
            }.issubset(
                balanced
            )

            for key in [
                "rmse",
                "mae",
                "mse",
            ]:

                value = float(
                    balanced[
                        key
                    ]
                )

                assert np.isfinite(
                    value
                )

                assert value >= 0


            by_dataset = sm[
                "by_dataset"
            ]

            assert len(
                by_dataset
            ) == 3

            assert {
                str(
                    row[
                        "dataset"
                    ]
                )
                for row in by_dataset
            } == DATASETS


            for row in by_dataset:

                rmse = float(
                    row[
                        "RMSE"
                    ]
                )

                mae = float(
                    row[
                        "MAE"
                    ]
                )

                assert np.isfinite(
                    rmse
                )

                assert np.isfinite(
                    mae
                )

                assert rmse >= 0
                assert mae >= 0


        assert np.isfinite(
            float(
                q[
                    "best_validation_balanced_rmse"
                ]
            )
        )

        assert np.isfinite(
            float(
                q[
                    "elapsed_seconds"
                ]
            )
        )


    assert (
        int(
            r[
                "models"
            ][
                "MLP"
            ][
                "parameter_count"
            ]
        )
        ==
        int(
            r[
                "models"
            ][
                "CAUSAL_GCN"
            ][
                "parameter_count"
            ]
        )
    )


    record_count += 1


assert record_count == 60

assert fit_count == 120

assert compatible_prefixed_runner_records == 3

assert set(
    data_counts_by_seed
) == {
    0, 1, 2, 3, 4
}


# ------------------------------------------------------------
# Cross-check protocol cardinality and interpretation barriers.
# ------------------------------------------------------------

assert p[
    "training_contract"
][
    "expected_model_fits"
] == 120

assert p[
    "training_contract"
][
    "new_hyperparameter_tuning"
] is False

assert p[
    "data_contract"
][
    "same_row_ids_all_four_conditions"
] is True

assert p[
    "interpretation_contract"
][
    "inferential_tests_added"
] is False

assert p[
    "interpretation_contract"
][
    "feature_dimension_difference_disclosed"
] is True

assert p[
    "interpretation_contract"
][
    "universal_gcn_superiority_claim_allowed"
] is False


print(
    "DQDV TRAINING EXECUTION AUDIT"
)

print(
    "protocol exact:",
    True
)

print(
    "runner exact:",
    True
)

print(
    "planned records:",
    60
)

print(
    "audited records:",
    record_count
)

print(
    "audited model fits:",
    fit_count
)

print(
    "all model-state hashes exact:",
    True
)

print(
    "compatible pre-input-binding records:",
    compatible_prefixed_runner_records
)

print(
    "legacy records restricted to valid 20D PRIMARY20_COMMON:",
    True
)

print(
    "same split row counts across conditions:",
    True
)

print(
    "MLP/GCN parameter matched within every condition:",
    True
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below epoch ceiling:",
    True
)

print(
    "all balanced and dataset metrics finite:",
    True
)

print(
    "test metrics parsed for independent audit:",
    True
)

print(
    "test metric numeric values surfaced before audit pass:",
    False
)

print(
    "test metrics used to change protocol:",
    False
)

print(
    "new hyperparameter tuning:",
    False
)

print(
    "new inferential tests:",
    False
)

print(
    "execution manifest SHA256:",
    sha256(
        MANIFEST
    )
)

print(
    "DQDV_TRAINING_EXECUTION_AUDIT_PASS=True"
)
