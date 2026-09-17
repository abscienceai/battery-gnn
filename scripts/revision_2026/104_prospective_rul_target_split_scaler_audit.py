from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


BASE = Path.home() / "alper/battery_revision_2026"

REPO = (
    BASE
    / "repo/battery-gnn"
)

ROOT = (
    BASE
    / "results/revision_2026/prospective_rul"
)

DATA = (
    ROOT
    / "data_v1"
)

PROTOCOL = (
    ROOT
    / "prospective_rul_protocol_v1.json"
)

LABELS = (
    BASE
    / "data/processed/revision_2026"
    / "unified_label_backbone.csv"
)

TASKS = (
    BASE
    / "data/processed/revision_2026/canonical_tasks"
    / "canonical_cycle_supervised_table.csv"
)

SOURCE_ASSIGN = (
    BASE
    / "data/processed/revision_2026/splits_and_scalers"
    / "cell_split_assignments.csv"
)

MANIFEST = (
    DATA
    / "prospective_rul_data_manifest.json"
)

TARGET = (
    DATA
    / "prospective_rul_target_table.csv"
)

ASSIGN = (
    DATA
    / "inherited_cell_split_assignments.csv"
)

FEATURE_SCALER = (
    DATA
    / "train_only_feature_scaler_parameters.csv"
)

TARGET_SCALER = (
    DATA
    / "train_only_target_scaler_parameters.csv"
)

SUMMARY = (
    DATA
    / "train_only_scaler_fit_summary.csv"
)

BUILDER = (
    REPO
    / "scripts/revision_2026"
    / "103_prospective_rul_target_split_scaler_builder.py"
)


EXPECTED_PROTOCOL_SHA = (
    "2f61fd329008286bca986411a88945c3ac0e5df09aeffab48939bdb92a9f4592"
)

EXPECTED_ASSIGN_SHA = (
    "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055"
)


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


def boolish(series):

    if series.dtype == bool:
        return series.astype(bool)

    out = (
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

    assert out.notna().all()

    return out.astype(bool)


def close(a, b):

    return math.isclose(
        float(a),
        float(b),
        rel_tol=1e-11,
        abs_tol=1e-11,
    )


assert sha256(
    PROTOCOL
) == EXPECTED_PROTOCOL_SHA

assert sha256(
    SOURCE_ASSIGN
) == EXPECTED_ASSIGN_SHA


m = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert m[
    "contract"
] == "prospective_rul_data_v1"

assert m[
    "status"
] == "COMPLETE_PENDING_INDEPENDENT_AUDIT"

assert m[
    "protocol_sha256"
] == EXPECTED_PROTOCOL_SHA

assert m[
    "source_split_assignment_sha256"
] == EXPECTED_ASSIGN_SHA

assert m[
    "builder"
][
    "sha256"
] == sha256(
    BUILDER
)


for item in m[
    "outputs"
].values():

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


target = pd.read_csv(
    TARGET,
    low_memory=False,
)

assign_out = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

feature_scaler = pd.read_csv(
    FEATURE_SCALER,
    low_memory=False,
)

target_scaler = pd.read_csv(
    TARGET_SCALER,
    low_memory=False,
)

summary = pd.read_csv(
    SUMMARY,
    low_memory=False,
)


# ------------------------------------------------------------
# Independently reconstruct harmonized k2 endpoint.
# ------------------------------------------------------------

labels = pd.read_csv(
    LABELS,
    low_memory=False,
)

labels = labels[
    labels[
        "dataset"
    ].isin(
        [
            "CALCE",
            "MATR",
        ]
    )
].copy()

labels[
    "observation_index"
] = pd.to_numeric(
    labels[
        "observation_index"
    ],
    errors="raise",
).astype(np.int64)

labels[
    "SOH"
] = pd.to_numeric(
    labels[
        "SOH"
    ],
    errors="coerce",
)


eol_map = {}


for (
    dataset,
    cell,
), g in labels.groupby(
    [
        "dataset",
        "cell_id",
    ],
    sort=False,
):

    g = g.sort_values(
        "observation_index",
        kind="stable",
    )

    obs = g[
        "observation_index"
    ].to_numpy(
        dtype=np.int64
    )

    soh = g[
        "SOH"
    ].to_numpy(
        dtype=np.float64
    )

    low = (
        np.isfinite(
            soh
        )
        & (
            soh <= 0.80
        )
    )

    first = None

    for i in range(
        len(low) - 1
    ):

        if not (
            bool(
                low[i]
            )
            and bool(
                low[i + 1]
            )
        ):
            continue

        if obs[
            i + 1
        ] != (
            obs[
                i
            ]
            + 1
        ):
            continue

        first = int(
            obs[
                i
            ]
        )

        break

    if first is not None:

        eol_map[
            (
                str(
                    dataset
                ),
                str(
                    cell
                ),
            )
        ] = first


assert sum(
    key[
        0
    ] == "CALCE"
    for key in eol_map
) == 13

assert sum(
    key[
        0
    ] == "MATR"
    for key in eol_map
) == 88


# ------------------------------------------------------------
# Reconstruct target rows independently.
# ------------------------------------------------------------

tasks = pd.read_csv(
    TASKS,
    low_memory=False,
)

tasks = tasks[
    tasks[
        "dataset"
    ].isin(
        [
            "CALCE",
            "MATR",
        ]
    )
].copy()

tasks[
    "feature_valid"
] = boolish(
    tasks[
        "feature_valid"
    ]
)

tasks[
    "observation_index"
] = pd.to_numeric(
    tasks[
        "observation_index"
    ],
    errors="raise",
).astype(np.int64)


tasks[
    "_key"
] = list(
    zip(
        tasks[
            "dataset"
        ].astype(str),
        tasks[
            "cell_id"
        ].astype(str),
    )
)

tasks[
    "_eol"
] = tasks[
    "_key"
].map(
    eol_map
)

expected = tasks[
    tasks[
        "feature_valid"
    ]
    & tasks[
        "_eol"
    ].notna()
    & (
        tasks[
            "observation_index"
        ]
        <= tasks[
            "_eol"
        ]
    )
].copy()


expected[
    "harmonized_eol_observation_index"
] = expected[
    "_eol"
].astype(
    np.int64
)

expected[
    "RUL_observations_80pct_k2"
] = (
    expected[
        "harmonized_eol_observation_index"
    ]
    - expected[
        "observation_index"
    ]
).astype(
    np.int64
)


expected = expected.sort_values(
    [
        "dataset",
        "cell_id",
        "observation_index",
    ],
    kind="stable",
).reset_index(
    drop=True
)


expected[
    "feature_valid_rank_one_based"
] = (
    expected.groupby(
        [
            "dataset",
            "cell_id",
        ],
        sort=False,
    )
    .cumcount()
    + 1
)


assert len(
    expected
) == 64169

assert len(
    target
) == 64169

assert int(
    (
        target[
            "dataset"
        ]
        == "CALCE"
    ).sum()
) == 7355

assert int(
    (
        target[
            "dataset"
        ]
        == "MATR"
    ).sum()
) == 56814


key_cols = [
    "dataset",
    "cell_id",
    "observation_index",
]

assert not target.duplicated(
    key_cols
).any()

assert list(
    target[
        key_cols
    ].itertuples(
        index=False,
        name=None,
    )
) == list(
    expected[
        key_cols
    ].itertuples(
        index=False,
        name=None,
    )
)


for col in [
    "harmonized_eol_observation_index",
    "RUL_observations_80pct_k2",
    "feature_valid_rank_one_based",
]:

    assert np.array_equal(
        target[
            col
        ].to_numpy(),
        expected[
            col
        ].to_numpy(),
    )


for feature in FEATURES:

    a = target[
        feature
    ].to_numpy(
        dtype=np.float64
    )

    b = expected[
        feature
    ].to_numpy(
        dtype=np.float64
    )

    # CSV serialization/deserialization can change the final
    # floating-point representation by sub-machine-scale decimal
    # round-trip differences. Identity columns and integer targets
    # remain exact; feature values are therefore checked with a
    # deliberately tight numerical tolerance.
    np.testing.assert_allclose(
        a,
        b,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )

    assert np.isfinite(
        a
    ).all()


assert (
    target[
        "RUL_observations_80pct_k2"
    ]
    >= 0
).all()

assert int(
    (
        target[
            "RUL_observations_80pct_k2"
        ]
        == 0
    ).sum()
) == 101

assert "HUST" not in set(
    target[
        "dataset"
    ]
)


# ------------------------------------------------------------
# Split assignments must be exact frozen subset.
# ------------------------------------------------------------

source_assign = pd.read_csv(
    SOURCE_ASSIGN,
    low_memory=False,
)

event_keys = set(
    zip(
        target[
            "dataset"
        ].astype(str),
        target[
            "cell_id"
        ].astype(str),
    )
)


expected_assign = source_assign[
    [
        (
            str(dataset),
            str(cell),
        )
        in event_keys
        for dataset, cell in zip(
            source_assign[
                "dataset"
            ],
            source_assign[
                "cell_id"
            ],
        )
    ]
].copy()


cols = list(
    assign_out.columns
)

expected_assign = expected_assign[
    cols
]

expected_assign = expected_assign.sort_values(
    [
        "split_seed",
        "dataset",
        "cell_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)

actual_assign = assign_out.sort_values(
    [
        "split_seed",
        "dataset",
        "cell_id",
    ],
    kind="stable",
).reset_index(
    drop=True
)


pd.testing.assert_frame_equal(
    actual_assign,
    expected_assign,
    check_dtype=False,
    check_exact=True,
)


assert len(
    actual_assign
) == 505


for seed in range(5):

    g = actual_assign[
        actual_assign[
            "split_seed"
        ]
        == seed
    ]

    counts = (
        g.groupby(
            [
                "dataset",
                "split",
            ]
        )
        .size()
        .to_dict()
    )

    assert counts == {
        (
            "CALCE",
            "test",
        ):
            2,

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
            "MATR",
            "test",
        ):
            13,

        (
            "MATR",
            "train",
        ):
            62,

        (
            "MATR",
            "validation",
        ):
            13,
    }


# ------------------------------------------------------------
# Independently reproduce all train-only scalers.
# ------------------------------------------------------------

assert len(
    feature_scaler
) == 100

assert len(
    target_scaler
) == 5

assert len(
    summary
) == 10


for seed in range(5):

    a = actual_assign[
        actual_assign[
            "split_seed"
        ]
        == seed
    ][
        [
            "dataset",
            "cell_id",
            "split",
        ]
    ]

    joined = target.merge(
        a,
        on=[
            "dataset",
            "cell_id",
        ],
        validate="many_to_one",
    )

    train = joined[
        joined[
            "split"
        ]
        == "train"
    ]


    assert train[
        [
            "dataset",
            "cell_id",
        ]
    ].drop_duplicates().shape[
        0
    ] == 71


    fs = feature_scaler[
        feature_scaler[
            "split_seed"
        ]
        == seed
    ]

    assert len(
        fs
    ) == 20

    assert set(
        fs[
            "feature"
        ]
    ) == set(
        FEATURES
    )

    assert set(
        fs[
            "fit_split"
        ]
    ) == {
        "train_only"
    }


    for feature in FEATURES:

        row = fs[
            fs[
                "feature"
            ]
            == feature
        ].iloc[
            0
        ]

        x = train[
            feature
        ].to_numpy(
            dtype=np.float64
        )

        mean = float(
            np.mean(
                x,
                dtype=np.float64,
            )
        )

        variance = float(
            np.var(
                x,
                ddof=0,
                dtype=np.float64,
            )
        )

        zero = bool(
            variance == 0.0
        )

        scale = (
            1.0
            if zero
            else math.sqrt(
                variance
            )
        )

        assert close(
            row[
                "mean"
            ],
            mean,
        )

        assert close(
            row[
                "variance"
            ],
            variance,
        )

        assert close(
            row[
                "scale"
            ],
            scale,
        )

        assert bool(
            row[
                "zero_variance"
            ]
        ) == zero

        assert int(
            row[
                "n_train_rows"
            ]
        ) == len(
            train
        )

        assert int(
            row[
                "n_train_cells"
            ]
        ) == 71


    ts = target_scaler[
        target_scaler[
            "split_seed"
        ]
        == seed
    ]

    assert len(
        ts
    ) == 1

    row = ts.iloc[
        0
    ]

    y = train[
        "RUL_observations_80pct_k2"
    ].to_numpy(
        dtype=np.float64
    )

    mean = float(
        np.mean(
            y,
            dtype=np.float64,
        )
    )

    variance = float(
        np.var(
            y,
            ddof=0,
            dtype=np.float64,
        )
    )

    zero = bool(
        variance == 0.0
    )

    scale = (
        1.0
        if zero
        else math.sqrt(
            variance
        )
    )

    assert close(
        row[
            "mean"
        ],
        mean,
    )

    assert close(
        row[
            "variance"
        ],
        variance,
    )

    assert close(
        row[
            "scale"
        ],
        scale,
    )

    assert bool(
        row[
            "zero_variance"
        ]
    ) == zero

    assert int(
        row[
            "n_train_rows"
        ]
    ) == len(
        train
    )

    assert int(
        row[
            "n_train_cells"
        ]
    ) == 71


# ------------------------------------------------------------
# Early prospective checkpoints exist by observed valid-history
# rank, never by fraction of future life.
# ------------------------------------------------------------

for k in [
    25,
    50,
    100,
]:

    z = target[
        target[
            "feature_valid_rank_one_based"
        ]
        == k
    ]

    assert len(
        z
    ) > 0

    assert not z.duplicated(
        [
            "dataset",
            "cell_id",
        ]
    ).any()


state = m[
    "execution_state"
]

assert state[
    "model_training_performed"
] is False

assert state[
    "test_metrics_read"
] is False

assert state[
    "test_metrics_surfaced"
] is False


print(
    "PROSPECTIVE RUL DATA AUDIT"
)

print(
    "harmonized k2 endpoint independently reconstructed:",
    True
)

print(
    "target rows exact:",
    64169
)

print(
    "CALCE rows exact:",
    7355
)

print(
    "MATR rows exact:",
    56814
)

print(
    "event cells exact:",
    101
)

print(
    "HUST exact-RUL rows:",
    0
)

print(
    "all EOL targets zero:",
    True
)

print(
    "no post-EOL targets:",
    True
)

print(
    "505 inherited assignment rows exact:",
    True
)

print(
    "per-seed cells train/validation/test:",
    "71/15/15"
)

print(
    "100 feature-scaler rows independently reproduced:",
    True
)

print(
    "5 target-scaler rows independently reproduced:",
    True
)

print(
    "all scalers train-only:",
    True
)

print(
    "K25/K50/K100 observed-history checkpoints materialized:",
    True
)

print(
    "model training performed:",
    False
)

print(
    "test metrics read:",
    False
)

print(
    "PROSPECTIVE_RUL_DATA_AUDIT_PASS=True"
)
