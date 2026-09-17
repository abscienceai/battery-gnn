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

OUTDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
)

SUP = (
    OUTDIR
    / "canonical_cycle_supervised_table.csv"
)

RHIST = (
    OUTDIR
    / "RWTH_usable_history_cycles.csv"
)

RTGT = (
    OUTDIR
    / "RWTH_sparse_target_table.csv"
)

MAN = (
    OUTDIR
    / "canonical_task_alignment_manifest.json"
)


EXPECTED_HASHES = {
    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "rwth_history":
        "4e8dcc9d80e0284e3c6cb9f912505fbff066efff052d3e08996da4028e3a0faa",

    "rwth_targets":
        "6346566441b265e9f5d328527ebeb58a0c9dfb2d18ba70ab72ea7b9f2808eb54",

    "manifest":
        "f0a9fe3f10c4adc05c3dd76947da824811cf42a5070261f0e14728dbd0b7acd9",
}


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


def as_bool(s):
    return (
        s
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "1.0",
                "yes",
            ]
        )
    )


print("=" * 80)
print("CANONICAL TASK ALIGNMENT INDEPENDENT AUDIT")
print("=" * 80)


# =========================================================================
# 1. Immutable artifact hashes
# =========================================================================

observed_hashes = {
    "supervised":
        sha256_file(SUP),

    "rwth_history":
        sha256_file(RHIST),

    "rwth_targets":
        sha256_file(RTGT),

    "manifest":
        sha256_file(MAN),
}

assert (
    observed_hashes
    == EXPECTED_HASHES
)

print(
    "artifact hashes exact: True"
)


# =========================================================================
# 2. Manifest self-consistency
# =========================================================================

manifest = json.loads(
    MAN.read_text(
        encoding="utf-8"
    )
)

assert (
    manifest[
        "split_applied"
    ]
    is False
)

assert (
    manifest[
        "scaler_applied"
    ]
    is False
)

assert (
    manifest[
        "target_like_primary_feature_names"
    ]
    == []
)

assert (
    manifest[
        "supervised_table_rows"
    ]
    == 300595
)

assert (
    manifest[
        "supervised_rows_by_dataset"
    ]
    == {
        "CALCE": 14298,
        "HUST": 146122,
        "MATR": 140175,
    }
)

assert (
    manifest[
        "feature_valid_by_dataset"
    ]
    == {
        "CALCE": 14298,
        "HUST": 146122,
        "MATR": 140133,
    }
)

assert (
    manifest[
        "rwth"
    ][
        "usable_qd_clean_contiguous_cycles"
    ]
    == 104940
)

assert (
    manifest[
        "rwth"
    ][
        "sparse_targets"
    ]
    == 777
)

assert (
    manifest[
        "rwth"
    ][
        "targets_with_usable_history"
    ]
    == 729
)

assert (
    manifest[
        "rwth"
    ][
        "zero_history_targets"
    ]
    == 48
)

assert (
    manifest[
        "rwth"
    ][
        "targets_whose_history_count_changes_vs_legacy"
    ]
    == 390
)

assert (
    manifest[
        "rwth"
    ][
        "targets_whose_latest_prior_cycle_changes_vs_legacy"
    ]
    == 0
)

print(
    "manifest contract exact: True"
)


# =========================================================================
# 3. Supervised-table metadata audit
# =========================================================================

meta_cols = [
    "dataset",
    "cell_id",
    "dataset_cell_key",
    "split_group",
    "observation_index",
    "native_cycle",
    "SOH",
    "RUL_observations",
    "valid_capacity",
    "feature_source_cell_id",
    "feature_raw_cycle_position_zero_based",
    "feature_available",
    "feature_valid",
    "supervised_feature_eligible",
    "soh_task_eligible",
    "rul_task_eligible",
    "feature_alignment_status",
]

s = pd.read_csv(
    SUP,
    usecols=meta_cols,
    low_memory=False,
)

assert len(s) == 300595

counts = (
    s[
        "dataset"
    ]
    .value_counts()
    .to_dict()
)

assert counts == {
    "HUST": 146122,
    "MATR": 140175,
    "CALCE": 14298,
}


assert not s[
    [
        "dataset_cell_key",
        "observation_index",
    ]
].duplicated().any()


available = (
    s.groupby(
        "dataset"
    )[
        "feature_available"
    ]
    .apply(
        lambda x:
            int(
                as_bool(x).sum()
            )
    )
    .to_dict()
)

valid = (
    s.groupby(
        "dataset"
    )[
        "feature_valid"
    ]
    .apply(
        lambda x:
            int(
                as_bool(x).sum()
            )
    )
    .to_dict()
)

eligible = (
    s.groupby(
        "dataset"
    )[
        "supervised_feature_eligible"
    ]
    .apply(
        lambda x:
            int(
                as_bool(x).sum()
            )
    )
    .to_dict()
)

soh = (
    s.groupby(
        "dataset"
    )[
        "soh_task_eligible"
    ]
    .apply(
        lambda x:
            int(
                as_bool(x).sum()
            )
    )
    .to_dict()
)

rul = (
    s.groupby(
        "dataset"
    )[
        "rul_task_eligible"
    ]
    .apply(
        lambda x:
            int(
                as_bool(x).sum()
            )
    )
    .to_dict()
)


assert available == {
    "CALCE": 14298,
    "HUST": 146122,
    "MATR": 140174,
}

assert valid == {
    "CALCE": 14298,
    "HUST": 146122,
    "MATR": 140133,
}

assert eligible == {
    "CALCE": 14298,
    "HUST": 146122,
    "MATR": 140133,
}

assert soh == {
    "CALCE": 14298,
    "HUST": 146122,
    "MATR": 140133,
}

assert rul == {
    "CALCE": 7355,
    "HUST": 146122,
    "MATR": 55211,
}


print(
    "supervised rows by dataset:",
    counts
)

print(
    "feature valid:",
    valid
)

print(
    "SOH eligible:",
    soh
)

print(
    "RUL eligible:",
    rul
)


# =========================================================================
# 4. HUST exact key semantics
# =========================================================================

h = (
    s[
        s["dataset"]
        == "HUST"
    ]
    .copy()
)

assert len(h) == 146122

assert as_bool(
    h[
        "feature_available"
    ]
).all()

assert (
    pd.to_numeric(
        h[
            "feature_raw_cycle_position_zero_based"
        ],
        errors="raise",
    )
    ==
    pd.to_numeric(
        h[
            "observation_index"
        ],
        errors="raise",
    )
).all()

assert (
    pd.to_numeric(
        h[
            "feature_raw_cycle_position_zero_based"
        ],
        errors="raise",
    )
    + 1
    ==
    pd.to_numeric(
        h[
            "native_cycle"
        ],
        errors="raise",
    )
).all()

print(
    "HUST exact alignment: 146122 / 146122"
)


# =========================================================================
# 5. CALCE exact retained cohort semantics
# =========================================================================

c = (
    s[
        s["dataset"]
        == "CALCE"
    ]
    .copy()
)

assert len(c) == 14298

assert as_bool(
    c[
        "feature_available"
    ]
).all()

assert as_bool(
    c[
        "feature_valid"
    ]
).all()

assert (
    pd.to_numeric(
        c[
            "feature_raw_cycle_position_zero_based"
        ],
        errors="raise",
    )
    ==
    pd.to_numeric(
        c[
            "native_cycle"
        ],
        errors="raise",
    )
).all()

print(
    "CALCE clean alignment: 14298 / 14298"
)


# =========================================================================
# 6. MATR exact edge cases
# =========================================================================

m = (
    s[
        s["dataset"]
        == "MATR"
    ]
    .copy()
)

assert len(m) == 140175

assert int(
    as_bool(
        m[
            "feature_available"
        ]
    ).sum()
) == 140174

assert int(
    as_bool(
        m[
            "feature_valid"
        ]
    ).sum()
) == 140133


unavailable = m[
    ~as_bool(
        m[
            "feature_available"
        ]
    )
]

assert len(
    unavailable
) == 1

assert (
    unavailable[
        "cell_id"
    ].iloc[0]
    == "b4c31"
)

assert pd.isna(
    unavailable[
        "native_cycle"
    ].iloc[0]
)

assert (
    unavailable[
        "feature_alignment_status"
    ].iloc[0]
    == "features_unavailable"
)


mapped_invalid = (
    as_bool(
        m[
            "feature_available"
        ]
    )
    &
    ~as_bool(
        m[
            "feature_valid"
        ]
    )
)

assert int(
    mapped_invalid.sum()
) == 41

assert set(
    m.loc[
        mapped_invalid,
        "feature_alignment_status",
    ].astype(str)
) == {
    "feature_invalid"
}


print(
    "MATR matched / valid / featureless:",
    "140174 / 140133 / 1"
)

print(
    "MATR mapped invalid:",
    41
)


# =========================================================================
# 7. All eligible primary features finite, chunked
# =========================================================================

eligible_rows_checked = 0

for chunk in pd.read_csv(
    SUP,
    usecols=[
        "supervised_feature_eligible",
    ] + FEATURES,
    chunksize=50000,
    low_memory=False,
):

    mask = as_bool(
        chunk[
            "supervised_feature_eligible"
        ]
    )

    x = (
        chunk.loc[
            mask,
            FEATURES,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    assert np.isfinite(
        x
    ).all()

    eligible_rows_checked += len(
        x
    )


assert (
    eligible_rows_checked
    == 300553
)

print(
    "eligible supervised rows all 20 finite:",
    eligible_rows_checked
)


# =========================================================================
# 8. RWTH usable-history table
# =========================================================================

rh = pd.read_csv(
    RHIST,
    usecols=[
        "canonical_cell_id",
        "usable_history_rank_zero_based",
        "cycle_number_pre_filter",
        "cycle_number_after_filter",
        "zyk_ts_last",
    ],
    low_memory=False,
)

assert len(rh) == 104940

assert (
    rh[
        "canonical_cell_id"
    ].nunique()
    == 48
)


for cell, g in rh.groupby(
    "canonical_cell_id",
    sort=False,
):
    rank = pd.to_numeric(
        g[
            "usable_history_rank_zero_based"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    assert np.array_equal(
        rank,
        np.arange(
            len(g)
        ),
    )

    ts = pd.to_numeric(
        g[
            "zyk_ts_last"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    assert np.all(
        np.diff(ts) >= 0
    )


rwth_feature_rows_checked = 0

for chunk in pd.read_csv(
    RHIST,
    usecols=FEATURES,
    chunksize=50000,
    low_memory=False,
):
    x = (
        chunk[
            FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    assert np.isfinite(
        x
    ).all()

    rwth_feature_rows_checked += len(
        x
    )


assert (
    rwth_feature_rows_checked
    == 104940
)

print(
    "RWTH usable history cycles:",
    len(rh)
)


# =========================================================================
# 9. RWTH target-specific causal prefix audit
# =========================================================================

rt = pd.read_csv(
    RTGT,
    low_memory=False,
)

assert len(rt) == 777

assert (
    rt[
        "cell_id"
    ].nunique()
    == 48
)

assert int(
    (
        pd.to_numeric(
            rt[
                "usable_history_count"
            ],
            errors="raise",
        )
        == 0
    ).sum()
) == 48

assert int(
    as_bool(
        rt[
            "has_usable_sensor_history"
        ]
    ).sum()
) == 729

assert int(
    (
        pd.to_numeric(
            rt[
                "removed_structural_boundary_cycles"
            ],
            errors="raise",
        )
        > 0
    ).sum()
) == 390

assert int(
    pd.to_numeric(
        rt[
            "removed_structural_boundary_cycles"
        ],
        errors="raise",
    ).sum()
) == 609

assert int(
    pd.to_numeric(
        rt[
            "removed_structural_boundary_cycles"
        ],
        errors="raise",
    ).max()
) == 8


history_by_cell = {
    cell:
        g.reset_index(
            drop=True
        )
    for cell, g in
    rh.groupby(
        "canonical_cell_id",
        sort=False,
    )
}


prefix_exact = 0

latest_exact = 0


for _, tr in rt.iterrows():

    cell = str(
        tr[
            "cell_id"
        ]
    )

    target_ts = int(
        tr[
            "target_ts"
        ]
    )

    count = int(
        tr[
            "usable_history_count"
        ]
    )

    g = history_by_cell[
        cell
    ]

    ts = pd.to_numeric(
        g[
            "zyk_ts_last"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    recomputed = int(
        np.searchsorted(
            ts,
            target_ts,
            side="left",
        )
    )

    assert recomputed == count

    assert int(
        tr[
            "usable_history_start_rank_zero_based"
        ]
    ) == 0

    assert int(
        tr[
            "usable_history_end_rank_exclusive"
        ]
    ) == count

    prefix_exact += 1

    if count == 0:
        assert pd.isna(
            tr[
                "latest_usable_zyk_ts"
            ]
        )

    else:
        latest = g.iloc[
            count - 1
        ]

        assert int(
            latest[
                "zyk_ts_last"
            ]
        ) < target_ts

        assert int(
            latest[
                "zyk_ts_last"
            ]
        ) == int(
            tr[
                "latest_usable_zyk_ts"
            ]
        )

        assert int(
            latest[
                "cycle_number_pre_filter"
            ]
        ) == int(
            tr[
                "latest_usable_cycle_pre_filter"
            ]
        )

        assert int(
            latest[
                "cycle_number_after_filter"
            ]
        ) == int(
            tr[
                "latest_usable_cycle_after_filter"
            ]
        )

        latest_exact += 1


assert prefix_exact == 777
assert latest_exact == 729

assert set(
    rt[
        "causal_alignment_rule"
    ].astype(str)
) == {
    "zyk_ts_last < target_ts"
}


print(
    "RWTH causal prefixes exact:",
    prefix_exact,
    "/ 777"
)

print(
    "RWTH latest usable cycle exact:",
    latest_exact,
    "/ 729"
)

print(
    "RWTH zero-history targets:",
    48
)

print(
    "RWTH targets with shortened history:",
    390
)

print(
    "RWTH removed history incidences:",
    609
)


# =========================================================================
# 10. Final leakage / preprocessing guards
# =========================================================================

assert all(
    c not in FEATURES
    for c in [
        "capacity_Ah",
        "SOH",
        "RUL_observations",
        "native_RUL",
        "native_cycle",
        "observation_index",
        "cycle_number_pre_filter",
        "cycle_number_after_filter",
        "life_coordinate",
    ]
)

assert len(
    FEATURES
) == 20


print()
print(
    "primary feature count:",
    len(FEATURES)
)

print(
    "target-like primary features:",
    0
)

print(
    "split applied:",
    False
)

print(
    "scaler applied:",
    False
)

print()
print(
    "CANONICAL_TASK_ALIGNMENT_AUDIT_PASS=True"
)
