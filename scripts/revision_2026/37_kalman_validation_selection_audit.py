from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

REPO = (
    BASE
    / "repo"
    / "battery-gnn"
)

SELECTOR = (
    REPO
    / "scripts"
    / "revision_2026"
    / "36_kalman_validation_only_selector.py"
)

ROOT = (
    BASE
    / "results"
    / "revision_2026"
    / "temporal_controls"
    / "kalman_validation_selection_v1"
)

PROTOCOL = (
    ROOT
    / "selection_protocol.json"
)

FIT_SUMMARY = (
    ROOT
    / "fit_summary.csv"
)

SPLIT_SUMMARY = (
    ROOT
    / "split_level_validation_summary.csv"
)

Q_SUMMARY = (
    ROOT
    / "q_validation_summary.csv"
)

SELECTED = (
    ROOT
    / "selected_kalman.json"
)

MANIFEST = (
    ROOT
    / "selection_manifest.json"
)


Q_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
]

SPLIT_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

TRAIN_SEEDS = [
    0,
    1,
    2,
]

PATIENCE = 30
MAX_EPOCHS = 5000
TIE_TOLERANCE = 1e-12


def sha256_file(path):
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


print("=" * 80)
print("KALMAN VALIDATION SELECTION INDEPENDENT AUDIT")
print("=" * 80)


# =============================================================================
# 1. Static leakage guard
# =============================================================================

source = SELECTOR.read_text(
    encoding="utf-8"
)


assert "final_v3" not in source
assert "metrics.csv" not in source


print(
    "selector references final_v3:",
    False
)

print(
    "selector references metrics.csv:",
    False
)


# =============================================================================
# 2. Protocol
# =============================================================================

protocol = json.loads(
    PROTOCOL.read_text(
        encoding="utf-8"
    )
)


assert protocol[
    "contract"
] == "kalman_validation_selection_v1"

assert protocol[
    "q_grid"
] == Q_GRID

assert protocol[
    "measurement_variance_r"
] == 1.0

assert protocol[
    "initial_variance"
] == 1.0

assert protocol[
    "fit_count"
] == 60

assert protocol[
    "max_epochs"
] == 5000

assert protocol[
    "patience"
] == 30

assert protocol[
    "test_rows_in_model_tensors"
] == 0

assert protocol[
    "test_metrics_read"
] is False

assert protocol[
    "final_test_result_root_accessed"
] is False


protocol_sha = sha256_file(
    PROTOCOL
)


print(
    "protocol frozen q candidates:",
    len(
        protocol[
            "q_grid"
        ]
    )
)

print(
    "protocol planned fits:",
    protocol[
        "fit_count"
    ]
)

print(
    "protocol test rows in model tensors:",
    0
)


# =============================================================================
# 3. Fit-summary structure
# =============================================================================

fit = pd.read_csv(
    FIT_SUMMARY,
    low_memory=False,
)


assert len(
    fit
) == 60


assert set(
    fit[
        "q"
    ].astype(float)
) == set(
    Q_GRID
)

assert set(
    fit[
        "split_seed"
    ].astype(int)
) == set(
    SPLIT_SEEDS
)

assert set(
    fit[
        "train_seed"
    ].astype(int)
) == set(
    TRAIN_SEEDS
)


assert not fit.duplicated(
    [
        "q",
        "split_seed",
        "train_seed",
    ]
).any()


assert np.isfinite(
    fit[
        "best_validation_rmse"
    ]
).all()


assert (
    fit[
        "epochs_after_best"
    ].astype(int)
    == PATIENCE
).all()


assert (
    fit[
        "stop_epoch"
    ].astype(int)
    < MAX_EPOCHS
).all()


assert (
    fit[
        "stopped_by_patience"
    ].astype(str).str.lower()
    == "true"
).all()


assert (
    fit[
        "hit_epoch_ceiling"
    ].astype(str).str.lower()
    == "false"
).all()


assert (
    fit[
        "test_rows_used"
    ].astype(int)
    == 0
).all()


assert (
    fit[
        "test_metrics_read"
    ].astype(str).str.lower()
    == "false"
).all()


assert (
    fit[
        "final_test_result_root_accessed"
    ].astype(str).str.lower()
    == "false"
).all()


print(
    "completed fit rows:",
    len(
        fit
    )
)

print(
    "unique q/split/train combinations:",
    60
)

print(
    "all fits patience-valid:",
    True
)

print(
    "all fits below 5000 ceiling:",
    True
)


# =============================================================================
# 4. Independently inspect all per-fit histories
# =============================================================================

history_count = 0


q_name = {
    0.001: "kalman_q0p001",
    0.01: "kalman_q0p01",
    0.1: "kalman_q0p1",
    1.0: "kalman_q1p0",
}


for q in Q_GRID:

    for split_seed in SPLIT_SEEDS:

        for train_seed in TRAIN_SEEDS:

            row = fit[
                (
                    np.isclose(
                        fit[
                            "q"
                        ].astype(float),
                        q,
                        rtol=0.0,
                        atol=0.0,
                    )
                )
                &
                (
                    fit[
                        "split_seed"
                    ].astype(int)
                    == split_seed
                )
                &
                (
                    fit[
                        "train_seed"
                    ].astype(int)
                    == train_seed
                )
            ]


            assert len(
                row
            ) == 1


            d = (
                ROOT
                / q_name[
                    q
                ]
                / (
                    f"split{split_seed}"
                    f"_train{train_seed}"
                )
            )


            fit_json = json.loads(
                (
                    d
                    / "fit.json"
                ).read_text(
                    encoding="utf-8"
                )
            )


            assert fit_json[
                "protocol_sha256"
            ] == protocol_sha


            h = pd.read_csv(
                d
                / "history.csv",
                low_memory=False,
            )


            assert len(
                h
            ) == int(
                fit_json[
                    "stop_epoch"
                ]
            )


            assert int(
                h.loc[
                    h[
                        "validation_balanced_rmse"
                    ].idxmin(),
                    "epoch",
                ]
            ) == int(
                fit_json[
                    "best_epoch"
                ]
            )


            assert (
                int(
                    fit_json[
                        "stop_epoch"
                    ]
                )
                - int(
                    fit_json[
                        "best_epoch"
                    ]
                )
                == 30
            )


            assert sha256_file(
                d
                / "history.csv"
            ) == fit_json[
                "history_sha256"
            ]


            history_count += 1


assert history_count == 60


print(
    "per-fit histories audited:",
    history_count,
    "/ 60"
)


# =============================================================================
# 5. Independently reproduce aggregation
# =============================================================================

re_split = (
    fit
    .groupby(
        [
            "q",
            "split_seed",
        ],
        as_index=False,
    )
    .agg(
        validation_rmse_mean_over_3_seeds=(
            "best_validation_rmse",
            "mean",
        ),
        validation_rmse_sd_over_3_seeds=(
            "best_validation_rmse",
            "std",
        ),
        max_stop_epoch=(
            "stop_epoch",
            "max",
        ),
        fit_count=(
            "train_seed",
            "count",
        ),
    )
    .sort_values(
        [
            "q",
            "split_seed",
        ]
    )
    .reset_index(
        drop=True
    )
)


stored_split = (
    pd.read_csv(
        SPLIT_SUMMARY,
        low_memory=False,
    )
    .sort_values(
        [
            "q",
            "split_seed",
        ]
    )
    .reset_index(
        drop=True
    )
)


pd.testing.assert_frame_equal(
    re_split,
    stored_split,
    check_exact=False,
    rtol=1e-11,
    atol=1e-12,
)


re_q = (
    re_split
    .groupby(
        "q",
        as_index=False,
    )
    .agg(
        validation_rmse_mean_over_5_splits=(
            "validation_rmse_mean_over_3_seeds",
            "mean",
        ),
        validation_rmse_sd_over_5_splits=(
            "validation_rmse_mean_over_3_seeds",
            "std",
        ),
        max_stop_epoch=(
            "max_stop_epoch",
            "max",
        ),
        split_count=(
            "split_seed",
            "count",
        ),
    )
    .sort_values(
        "q"
    )
    .reset_index(
        drop=True
    )
)


stored_q = (
    pd.read_csv(
        Q_SUMMARY,
        low_memory=False,
    )
    .sort_values(
        "q"
    )
    .reset_index(
        drop=True
    )
)


for col in (
    "validation_rmse_mean_over_5_splits",
    "validation_rmse_sd_over_5_splits",
    "max_stop_epoch",
    "split_count",
):

    assert np.allclose(
        re_q[
            col
        ].to_numpy(
            dtype=float
        ),
        stored_q[
            col
        ].to_numpy(
            dtype=float
        ),
        rtol=1e-11,
        atol=1e-12,
    )


minimum = float(
    re_q[
        "validation_rmse_mean_over_5_splits"
    ].min()
)


tied = re_q[
    re_q[
        "validation_rmse_mean_over_5_splits"
    ]
    <= (
        minimum
        + TIE_TOLERANCE
    )
]


re_selected_q = float(
    tied[
        "q"
    ].min()
)


selected = json.loads(
    SELECTED.read_text(
        encoding="utf-8"
    )
)


assert np.isclose(
    selected[
        "selected_q"
    ],
    re_selected_q,
    rtol=0.0,
    atol=0.0,
)


assert selected[
    "test_rows_used"
] == 0

assert selected[
    "test_metrics_read"
] is False

assert selected[
    "final_test_result_root_accessed"
] is False


print(
    "split-level aggregation independently reproduced:",
    True
)

print(
    "q-level aggregation independently reproduced:",
    True
)

print(
    "tie rule independently reproduced:",
    True
)


# =============================================================================
# 6. Manifest
# =============================================================================

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)


assert manifest[
    "completed_fit_count"
] == 60

assert manifest[
    "all_fits_patience_valid"
] is True

assert manifest[
    "all_fits_below_epoch_ceiling"
] is True

assert manifest[
    "test_rows_used"
] == 0

assert manifest[
    "test_metrics_read"
] is False

assert manifest[
    "final_test_result_root_accessed"
] is False

assert np.isclose(
    manifest[
        "selected_q"
    ],
    re_selected_q,
    rtol=0.0,
    atol=0.0,
)


print()
print(
    "selected q:",
    re_selected_q
)

print(
    "selected r:",
    selected[
        "selected_r"
    ]
)

print(
    "test rows used:",
    0
)

print(
    "test metrics read:",
    False
)

print(
    "final test result root accessed:",
    False
)

print()
print(
    "KALMAN_VALIDATION_SELECTION_AUDIT_PASS=True"
)
