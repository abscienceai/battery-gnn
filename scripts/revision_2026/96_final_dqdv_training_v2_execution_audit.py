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

TRAINROOT = ROOT / "training_v2"
EXEC = TRAINROOT / "final_execution_v2"

PROTOCOL = TRAINROOT / "dqdv_training_protocol_v2.json"

COMMON_MANIFEST = (
    ROOT
    / "common_cohort_scalers_v1/common_cohort_scaler_manifest.json"
)

MANIFEST = EXEC / "execution_manifest.json"

STATE_DIR = EXEC / "model_states"

RUNNER = (
    REPO
    / "scripts/revision_2026/95_final_dqdv_training_v2_runner.py"
)

EXPECTED_PROTOCOL_SHA = (
    "4612a450eeaf909a04c31be16957bdc0567aac67635241e14d7bbceea507fe9d"
)

EXPECTED_COMMON_MANIFEST_SHA = (
    "3ee48af31ad580c14de0460fa1f163ad572e369368d6da71c0ba6823d28905a1"
)

COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA = (
    "46f1ad8ab073e320b5ca5420283cec15c5b2d661f3f2089b41716a5b4d72a76d"
)

PARENT_PROTOCOL_V1_SHA = (
    "43122c4d6ef5a2e70b2d13ae22432a2cfa13027874e0bea6f3b62675e3fb23e9"
)

COMPATIBLE_V1_INPUT_BINDING_RUNNER_SHA = (
    "117f5dea393cb27c487e07e7a0ffc7c46195bb71d69c7809fcc574517caa573f"
)

IMPORT_MANIFEST = (
    EXEC
    / "v1_compatible_import_manifest.json"
)

EXPECTED_IMPORT_MANIFEST_SHA = (
    "068223da9e12c370af136343388285b9930ee3edd42b14082309eb40a0c65117"
)

EXPECTED_EXECUTION_MANIFEST_SHA = (
    "744ae4a695f9a94a2c597469dc867c58141ea94fc5d131074d88574a25b72725"
)

PRECORRECTION_MANIFEST = (
    EXEC
    / "execution_manifest.pre_metadata_correction.json"
)

METADATA_CORRECTION_RECEIPT = (
    EXEC
    / "execution_manifest_metadata_correction.json"
)

EXPECTED_PRECORRECTION_MANIFEST_SHA = (
    "a9b74e84eb145cb5b2340ab40c036821d15d82d2ba2ce2caf1a93f490fa4f4a9"
)

EXPECTED_METADATA_CORRECTION_RECEIPT_SHA = (
    "ebc46e8d1eb265e8850f42c9af30f91246f3ccb56f84cb86a4f5a5a9cd1ad343"
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

assert sha256(
    MANIFEST
) == EXPECTED_EXECUTION_MANIFEST_SHA

assert sha256(
    PRECORRECTION_MANIFEST
) == EXPECTED_PRECORRECTION_MANIFEST_SHA

assert sha256(
    METADATA_CORRECTION_RECEIPT
) == EXPECTED_METADATA_CORRECTION_RECEIPT_SHA

metadata_correction = json.loads(
    METADATA_CORRECTION_RECEIPT.read_text(
        encoding="utf-8"
    )
)

assert metadata_correction[
    "old_execution_manifest_sha256"
] == EXPECTED_PRECORRECTION_MANIFEST_SHA

assert metadata_correction[
    "corrected_execution_manifest_sha256"
] == EXPECTED_EXECUTION_MANIFEST_SHA

assert metadata_correction[
    "corrected_field"
] == "epoch_ceiling"

assert metadata_correction[
    "old_value"
] == 5000

assert metadata_correction[
    "new_value"
] == 10000

assert metadata_correction[
    "scientific_execution_changed"
] is False

assert metadata_correction[
    "pair_records_changed"
] is False

assert metadata_correction[
    "model_states_changed"
] is False

assert metadata_correction[
    "protocol_changed"
] is False

assert metadata_correction[
    "runner_changed"
] is False

assert metadata_correction[
    "test_metric_values_read_for_correction"
] is False

assert metadata_correction[
    "test_metric_values_surfaced"
] is False

assert sha256(
    IMPORT_MANIFEST
) == EXPECTED_IMPORT_MANIFEST_SHA

imp = json.loads(
    IMPORT_MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert imp[
    "protocol_v2_sha256"
] == EXPECTED_PROTOCOL_SHA

assert imp[
    "parent_protocol_v1_sha256"
] == PARENT_PROTOCOL_V1_SHA

assert imp[
    "imported_pair_count"
] == 4

assert imp[
    "imported_model_fit_count"
] == 8

assert imp[
    "state_reference_count"
] == 8

assert imp[
    "record_bytes_preserved_exactly"
] is True

assert imp[
    "model_states_copied"
] is False

assert imp[
    "model_states_referenced_from_v1"
] is True

assert imp[
    "invalid_partial_v1_state_count"
] == 2

assert imp[
    "invalid_partial_v1_states_imported"
] is False

assert imp[
    "invalid_partial_v1_states_reused"
] is False

assert imp[
    "test_metric_values_inspected"
] is False

assert imp[
    "test_metric_values_surfaced"
] is False


imported_by_key = {}

for item in imp[
    "imported_records"
]:

    key = (
        str(
            item[
                "condition"
            ]
        ),
        int(
            item[
                "split_seed"
            ]
        ),
        int(
            item[
                "train_seed"
            ]
        ),
    )

    assert key not in imported_by_key

    imported_by_key[
        key
    ] = item


assert set(
    imported_by_key
) == {
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


assert m[
    "contract"
] == "dqdv_training_execution_v2"

assert m[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "common_cohort_manifest_sha256"
] == EXPECTED_COMMON_MANIFEST_SHA

assert m[
    "parent_protocol_v1_sha256"
] == PARENT_PROTOCOL_V1_SHA

assert m[
    "compatible_v1_import_manifest_sha256"
] == EXPECTED_IMPORT_MANIFEST_SHA

assert m[
    "compatible_v1_imported_pair_count"
] == 4

assert m[
    "compatible_v1_imported_model_fit_count"
] == 8

assert m[
    "compatible_v1_records_retrained"
] is False

assert m[
    "invalid_partial_v1_states_reused"
] is False

assert m[
    "validation_safety_ceiling"
] == 10000

assert m[
    "validation_patience"
] == 30

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
] == 10000

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

imported_record_count = 0
native_v2_record_count = 0

imported_fit_count = 0
native_v2_fit_count = 0


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
        "condition"
    ] == condition

    pair_key = (
        str(
            condition
        ),
        int(
            split_seed
        ),
        int(
            train_seed
        ),
    )

    imported_item = imported_by_key.get(
        pair_key
    )

    record_runner_sha = str(
        r[
            "runner_sha256"
        ]
    )


    if imported_item is not None:

        imported_record_count += 1

        assert r[
            "protocol_sha256"
        ] == PARENT_PROTOCOL_V1_SHA


        source_record = Path(
            imported_item[
                "source_record_path"
            ]
        )

        destination_record = Path(
            imported_item[
                "destination_record_path"
            ]
        )

        assert source_record.is_file()
        assert destination_record.is_file()

        assert rp.resolve() == destination_record.resolve()

        assert sha256(
            source_record
        ) == imported_item[
            "source_record_sha256"
        ]

        assert sha256(
            destination_record
        ) == imported_item[
            "destination_record_sha256"
        ]

        assert (
            imported_item[
                "source_record_sha256"
            ]
            ==
            imported_item[
                "destination_record_sha256"
            ]
        )

        assert source_record.read_bytes() == destination_record.read_bytes()

        assert imported_item[
            "record_bytes_preserved_exactly"
        ] is True


        if condition == "PRIMARY20_COMMON":

            assert split_seed == 0

            assert train_seed in {
                0,
                1,
                2,
            }

            assert (
                record_runner_sha
                == COMPATIBLE_PRE_INPUT_BINDING_RUNNER_SHA
            )

            assert int(
                r[
                    "input_dimension"
                ]
            ) == 20

            compatible_prefixed_runner_records += 1

        else:

            assert pair_key == (
                "PRIMARY20_PLUS_FULL_DQDV10",
                0,
                0,
            )

            assert (
                record_runner_sha
                == COMPATIBLE_V1_INPUT_BINDING_RUNNER_SHA
            )


    else:

        native_v2_record_count += 1

        assert r[
            "protocol_sha256"
        ] == EXPECTED_PROTOCOL_SHA

        assert (
            record_runner_sha
            == m[
                "runner_sha256"
            ]
        )

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


        if imported_item is not None:

            imported_fit_count += 1

            ref = imported_item[
                "state_references"
            ][
                model_name
            ]

            assert ref[
                "copied_into_v2_namespace"
            ] is False

            assert ref[
                "immutable_v1_reference"
            ] is True

            ref_path = Path(
                ref[
                    "source_path"
                ]
            )

            assert ref_path.is_file()

            assert sha256(
                ref_path
            ) == ref[
                "sha256"
            ]

            assert Path(
                q[
                    "state_path"
                ]
            ).resolve() == ref_path.resolve()

            assert q[
                "state_sha256"
            ] == ref[
                "sha256"
            ]


        else:

            native_v2_fit_count += 1

            state_path_native = Path(
                q[
                    "state_path"
                ]
            )

            assert state_path_native.is_file()

            assert (
                state_path_native.resolve().parent
                == STATE_DIR.resolve()
            )


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

        assert stop_epoch < 10000

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

assert imported_record_count == 4

assert native_v2_record_count == 56

assert imported_fit_count == 8

assert native_v2_fit_count == 112

assert compatible_prefixed_runner_records == 3


local_v2_states = sorted(
    STATE_DIR.glob(
        "*.pt"
    )
)

assert len(
    local_v2_states
) == 112

assert (
    len(
        local_v2_states
    )
    +
    int(
        imp[
            "state_reference_count"
        ]
    )
) == 120


# Invalid v1 partial states remain provenance-only and cannot appear
# in any completed record.
completed_state_paths = set()

for summary in m[
    "records"
]:

    rr = json.loads(
        Path(
            summary[
                "record_path"
            ]
        ).read_text(
            encoding="utf-8"
        )
    )

    for model_name in [
        "MLP",
        "CAUSAL_GCN",
    ]:

        completed_state_paths.add(
            str(
                Path(
                    rr[
                        "models"
                    ][
                        model_name
                    ][
                        "state_path"
                    ]
                ).resolve()
            )
        )


protocol_v2 = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)

invalid_partial = protocol_v2[
    "ceiling_amendment"
][
    "invalid_partial_v1_states"
]

assert len(
    invalid_partial
) == 2

for item in invalid_partial:

    assert item[
        "reusable_in_v2"
    ] is False

    invalid_path = Path(
        item[
            "path"
        ]
    )

    assert invalid_path.is_file()

    assert sha256(
        invalid_path
    ) == item[
        "sha256"
    ]

    assert str(
        invalid_path.resolve()
    ) not in completed_state_paths

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
    "DQDV_TRAINING_V2_EXECUTION_AUDIT_PASS=True"
)
