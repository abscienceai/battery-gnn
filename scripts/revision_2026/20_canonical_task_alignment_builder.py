from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd


# =============================================================================
# Paths / immutable contracts
# =============================================================================

BASE = Path(
    os.environ.get(
        "BATTERY_REVISION_BASE",
        Path.home()
        / "alper"
        / "battery_revision_2026",
    )
)

FEATURE_DIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "raw_cycle_features"
)

LABEL_BACKBONE = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "unified_label_backbone.csv"
)

AUDIT_DIR = BASE / "data" / "audit"

OUTDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUPERVISED_OUT = (
    OUTDIR
    / "canonical_cycle_supervised_table.csv"
)

RWTH_HISTORY_OUT = (
    OUTDIR
    / "RWTH_usable_history_cycles.csv"
)

RWTH_TARGET_OUT = (
    OUTDIR
    / "RWTH_sparse_target_table.csv"
)

MANIFEST_OUT = (
    OUTDIR
    / "canonical_task_alignment_manifest.json"
)


BUILDER_CONTRACT_VERSION = (
    "canonical_task_alignment_v1"
)

FEATURE_CONTRACT_VERSION = (
    "primary_vit20_v1"
)


EXPECTED_FEATURE_HASHES = {
    "HUST":
        "533f1d3d47b28bd819dec9822123bf0a80e6a9372e2e6d78b01b1a07c9ef1130",

    "CALCE":
        "83f35417c58ba3cd7615ba9593c59e9807be84da4ec3235e29a1ca786b9ef2d0",

    "MATR":
        "a52032a187a75ece8fecf455131c1e1c207ce574d766d88ac534f3bb13b96c4a",

    "RWTH":
        "f3cbc1463b17b394816a394de5e48b8f363caaea65a94c1567820ff3835046fe",
}


EXPECTED_LABEL_HASH = (
    "2a9d8b59c8a96dc19fe41d2be05142c1fe9ed2fa81530158aa0c9079e3d74834"
)


FEATURE_COLUMNS = [
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


CONTINUATIONS = {
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}


# =============================================================================
# Helpers
# =============================================================================

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


def as_bool(series):
    s = (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return s.isin(
        [
            "true",
            "1",
            "1.0",
            "yes",
        ]
    )


def cell_num(value):
    m = re.search(
        r"(\d+)$",
        str(value),
    )

    if not m:
        raise ValueError(
            f"Cannot parse RWTH cell number: {value}"
        )

    return int(
        m.group(1)
    )


def ts_num(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    try:
        return int(
            float(s)
        )
    except Exception:
        pass

    m = re.search(
        r"TS0*(\d+)",
        s,
        flags=re.IGNORECASE,
    )

    if not m:
        raise ValueError(
            f"Cannot parse RWTH TS identifier: {value}"
        )

    return int(
        m.group(1)
    )


def write_csv(df, path):
    df.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def load_feature(dataset):
    path = (
        FEATURE_DIR
        / f"{dataset}_raw_cycle_features.csv"
    )

    assert path.is_file(), path

    observed = sha256_file(
        path
    )

    assert (
        observed
        == EXPECTED_FEATURE_HASHES[dataset]
    ), (
        dataset,
        observed,
        EXPECTED_FEATURE_HASHES[dataset],
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    assert all(
        c in df.columns
        for c in FEATURE_COLUMNS
    )

    found = [
        c
        for c in df.columns
        if (
            c.startswith("v_")
            or c.startswith("i_")
        )
    ]

    assert found == FEATURE_COLUMNS, (
        dataset,
        found,
    )

    if "feature_contract_version" in df.columns:
        assert set(
            df[
                "feature_contract_version"
            ]
            .dropna()
            .astype(str)
            .unique()
        ) == {
            FEATURE_CONTRACT_VERSION
        }

    return df


def feature_projection(df):
    wanted = [
        "raw_cell_id",
        "source_member",
        "raw_cycle_key",
        "raw_cycle_position_zero_based",
        "feature_contract_version",
        "kernel_sha256",
        "feature_valid",
        "failure_reason",
    ] + FEATURE_COLUMNS

    out = (
        df[
            [
                c
                for c in wanted
                if c in df.columns
            ]
        ]
        .copy()
    )

    out = out.rename(
        columns={
            "raw_cell_id":
                "feature_source_cell_id",

            "source_member":
                "feature_source_member",

            "raw_cycle_key":
                "feature_raw_cycle_key",

            "raw_cycle_position_zero_based":
                "feature_raw_cycle_position_zero_based",

            "feature_contract_version":
                "feature_contract_version",

            "kernel_sha256":
                "feature_kernel_sha256",

            "feature_valid":
                "feature_valid_raw",

            "failure_reason":
                "feature_failure_reason",
        }
    )

    return out


def finalize_supervised_frame(df):
    df = df.copy()

    df["feature_available"] = (
        df[
            "feature_source_cell_id"
        ].notna()
    )

    df["feature_valid"] = (
        df["feature_available"]
        & as_bool(
            df[
                "feature_valid_raw"
            ]
        )
    )

    valid_capacity = as_bool(
        df[
            "valid_capacity"
        ]
    )

    df[
        "supervised_feature_eligible"
    ] = (
        df["feature_available"]
        & df["feature_valid"]
        & valid_capacity
    )

    df[
        "soh_task_eligible"
    ] = (
        df[
            "supervised_feature_eligible"
        ]
        & pd.to_numeric(
            df["SOH"],
            errors="coerce",
        ).notna()
    )

    df[
        "rul_task_eligible"
    ] = (
        df[
            "supervised_feature_eligible"
        ]
        & pd.to_numeric(
            df[
                "RUL_observations"
            ],
            errors="coerce",
        ).notna()
    )

    status = np.full(
        len(df),
        "feature_valid",
        dtype=object,
    )

    status[
        ~df[
            "feature_available"
        ].to_numpy()
    ] = (
        "features_unavailable"
    )

    status[
        (
            df["feature_available"]
            & ~df["feature_valid"]
        ).to_numpy()
    ] = (
        "feature_invalid"
    )

    df[
        "feature_alignment_status"
    ] = status

    return df


# =============================================================================
# Input guards
# =============================================================================

assert sha256_file(
    LABEL_BACKBONE
) == EXPECTED_LABEL_HASH

labels = pd.read_csv(
    LABEL_BACKBONE,
    low_memory=False,
)

assert len(labels) == 301372
assert (
    labels[
        "dataset_cell_key"
    ].nunique()
    == 307
)

assert labels[
    "dataset"
].value_counts().to_dict() == {
    "HUST": 146122,
    "MATR": 140175,
    "CALCE": 14298,
    "RWTH": 777,
}


# =============================================================================
# HUST
# =============================================================================

hf = load_feature(
    "HUST"
)

hl = (
    labels[
        labels["dataset"]
        == "HUST"
    ]
    .copy()
)

hp = feature_projection(
    hf
)

h = hl.merge(
    hp,
    left_on=[
        "cell_id",
        "observation_index",
    ],
    right_on=[
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert len(h) == 146122
assert (
    h["_merge"] == "both"
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
        h["native_cycle"],
        errors="raise",
    )
).all()

h = h.drop(
    columns="_merge"
)

h = finalize_supervised_frame(
    h
)

assert h[
    "supervised_feature_eligible"
].all()


# =============================================================================
# CALCE
# =============================================================================

cf = load_feature(
    "CALCE"
)

cl = (
    labels[
        labels["dataset"]
        == "CALCE"
    ]
    .copy()
)

audit_parts = []

for p in sorted(
    AUDIT_DIR.glob(
        "C[XS]2_*_cycles_audit.csv"
    )
):
    cell = p.name.replace(
        "_cycles_audit.csv",
        "",
    )

    x = pd.read_csv(
        p,
        usecols=[
            "batteryml_cycle_number_zero_based",
            "keep_final_batteryml",
        ],
        low_memory=False,
    )

    x["cell_id"] = cell

    audit_parts.append(
        x
    )

ca = pd.concat(
    audit_parts,
    ignore_index=True,
)

assert len(ca) == 18820
assert ca[
    "cell_id"
].nunique() == 13

cf_a = cf.merge(
    ca,
    left_on=[
        "raw_cell_id",
        "raw_cycle_position_zero_based",
    ],
    right_on=[
        "cell_id",
        "batteryml_cycle_number_zero_based",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert (
    cf_a["_merge"] == "both"
).all()

clean = as_bool(
    cf_a[
        "keep_final_batteryml"
    ]
)

assert int(
    clean.sum()
) == 14298

cf_clean = (
    cf_a[
        clean
    ]
    .drop(
        columns=[
            "_merge",
            "cell_id",
            "batteryml_cycle_number_zero_based",
            "keep_final_batteryml",
        ]
    )
    .copy()
)

cp = feature_projection(
    cf_clean
)

c = cl.merge(
    cp,
    left_on=[
        "cell_id",
        "native_cycle",
    ],
    right_on=[
        "feature_source_cell_id",
        "feature_raw_cycle_position_zero_based",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert len(c) == 14298
assert (
    c["_merge"] == "both"
).all()

c = c.drop(
    columns="_merge"
)

c = finalize_supervised_frame(
    c
)

assert c[
    "supervised_feature_eligible"
].all()


# =============================================================================
# MATR
# =============================================================================

mf = load_feature(
    "MATR"
)

ml = (
    labels[
        labels["dataset"]
        == "MATR"
    ]
    .copy()
)

feature_cells = set(
    mf[
        "raw_cell_id"
    ].astype(str)
)

canonical_cells = set(
    ml[
        "cell_id"
    ].astype(str)
)

donor_cells = {
    donor
    for donor, _ in
    CONTINUATIONS.values()
}

excluded_cells = sorted(
    feature_cells
    - canonical_cells
    - donor_cells
)

assert len(feature_cells) == 185
assert len(canonical_cells) == 169
assert len(donor_cells) == 5
assert len(excluded_cells) == 11

mapping_parts = []


for cell in sorted(
    canonical_cells
):
    p = (
        mf[
            mf[
                "raw_cell_id"
            ].astype(str)
            == cell
        ]
        .sort_values(
            "raw_cycle_position_zero_based"
        )
        .reset_index(
            drop=True
        )
    )

    assert len(p) > 0

    assert np.array_equal(
        pd.to_numeric(
            p[
                "raw_cycle_position_zero_based"
            ],
            errors="raise",
        ).to_numpy(
            dtype=int
        ),
        np.arange(
            len(p)
        ),
    )

    if cell in CONTINUATIONS:
        donor, expected_n = (
            CONTINUATIONS[cell]
        )

        d = (
            mf[
                mf[
                    "raw_cell_id"
                ].astype(str)
                == donor
            ]
            .sort_values(
                "raw_cycle_position_zero_based"
            )
            .reset_index(
                drop=True
            )
        )

        assert len(d) == expected_n

        assert np.array_equal(
            pd.to_numeric(
                d[
                    "raw_cycle_position_zero_based"
                ],
                errors="raise",
            ).to_numpy(
                dtype=int
            ),
            np.arange(
                len(d)
            ),
        )

        source = pd.concat(
            [
                p,
                d,
            ],
            ignore_index=True,
        )

    elif cell == "b4c31":
        source = p

    else:
        source = p

    source = feature_projection(
        source
    )

    source[
        "cell_id"
    ] = cell

    source[
        "observation_index"
    ] = np.arange(
        len(source),
        dtype=int,
    )

    mapping_parts.append(
        source
    )


mmap = pd.concat(
    mapping_parts,
    ignore_index=True,
)

assert len(mmap) == 140174

assert not mmap[
    [
        "cell_id",
        "observation_index",
    ]
].duplicated().any()


m = ml.merge(
    mmap,
    on=[
        "cell_id",
        "observation_index",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert len(m) == 140175

assert int(
    (
        m["_merge"]
        == "both"
    ).sum()
) == 140174

assert int(
    (
        m["_merge"]
        == "left_only"
    ).sum()
) == 1


unmatched = m[
    m["_merge"]
    == "left_only"
]

assert len(unmatched) == 1

assert (
    unmatched[
        "cell_id"
    ].iloc[0]
    == "b4c31"
)

assert pd.isna(
    unmatched[
        "native_cycle"
    ].iloc[0]
)

m = m.drop(
    columns="_merge"
)

m = finalize_supervised_frame(
    m
)

mapped_invalid = int(
    (
        m[
            "feature_available"
        ]
        & ~m[
            "feature_valid"
        ]
    ).sum()
)

assert mapped_invalid == 41

assert int(
    m[
        "feature_available"
    ].sum()
) == 140174

assert int(
    (
        m[
            "feature_available"
        ]
        & m[
            "feature_valid"
        ]
    ).sum()
) == 140133


excluded_mask = (
    mf[
        "raw_cell_id"
    ]
    .astype(str)
    .isin(
        excluded_cells
    )
)

assert int(
    excluded_mask.sum()
) == 14237

assert int(
    (
        ~as_bool(
            mf[
                "feature_valid"
            ]
        )
        & excluded_mask
    ).sum()
) == 5


# =============================================================================
# Canonical supervised table: HUST + CALCE + MATR
# =============================================================================

supervised = pd.concat(
    [
        h,
        c,
        m,
    ],
    ignore_index=True,
    sort=False,
)

dataset_order = {
    "HUST": 0,
    "CALCE": 1,
    "MATR": 2,
}

supervised[
    "_dataset_order"
] = supervised[
    "dataset"
].map(
    dataset_order
)

supervised = (
    supervised
    .sort_values(
        [
            "_dataset_order",
            "cell_id",
            "observation_index",
        ]
    )
    .drop(
        columns="_dataset_order"
    )
    .reset_index(
        drop=True
    )
)

assert len(
    supervised
) == 300595

assert supervised[
    "dataset"
].value_counts().to_dict() == {
    "HUST": 146122,
    "MATR": 140175,
    "CALCE": 14298,
}

assert not any(
    x in FEATURE_COLUMNS
    for x in (
        "capacity_Ah",
        "SOH",
        "RUL_observations",
        "native_RUL",
        "native_cycle",
        "observation_index",
    )
)

write_csv(
    supervised,
    SUPERVISED_OUT,
)


# =============================================================================
# RWTH usable causal history
# =============================================================================

rf = load_feature(
    "RWTH"
)

rr = pd.read_csv(
    AUDIT_DIR
    / "RWTH_ZYK_cycle_reconstruction_all_cells.csv",
    usecols=[
        "cell_id",
        "cycle_number_pre_filter",
        "cycle_number_after_filter",
        "batteryml_abnormal_removed",
    ],
    low_memory=False,
)

assert len(rr) == 105663


rf["_cell_num"] = (
    rf[
        "raw_cell_id"
    ].map(
        cell_num
    )
)

rf[
    "cycle_number_pre_filter"
] = (
    pd.to_numeric(
        rf[
            "raw_cycle_position_zero_based"
        ],
        errors="raise",
    )
    .astype(int)
    + 1
)

rf[
    "zyk_ts_last"
] = (
    rf[
        "source_zip_last"
    ].map(
        ts_num
    )
)


rr["_cell_num"] = (
    rr[
        "cell_id"
    ].map(
        cell_num
    )
)

rr[
    "cycle_number_pre_filter"
] = pd.to_numeric(
    rr[
        "cycle_number_pre_filter"
    ],
    errors="raise",
).astype(int)


rz = rf.merge(
    rr,
    on=[
        "_cell_num",
        "cycle_number_pre_filter",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=(
        "_feature",
        "_audit",
    ),
)

assert len(rz) == 105663
assert (
    rz["_merge"]
    == "both"
).all()

rz[
    "_feature_valid"
] = as_bool(
    rz[
        "feature_valid"
    ]
)

rz[
    "_qd_eligible"
] = rz[
    "cycle_number_after_filter"
].notna()

rz[
    "_usable"
] = (
    rz[
        "_feature_valid"
    ]
    & rz[
        "_qd_eligible"
    ]
)

assert int(
    rz[
        "_qd_eligible"
    ].sum()
) == 105006

assert int(
    rz[
        "_feature_valid"
    ].sum()
) == 105017

assert int(
    rz[
        "_usable"
    ].sum()
) == 104940

assert int(
    (
        ~rz[
            "_feature_valid"
        ]
        & rz[
            "_qd_eligible"
        ]
    ).sum()
) == 66


history = (
    rz[
        rz[
            "_usable"
        ]
    ]
    .copy()
)

history[
    "canonical_cell_id"
] = history[
    "_cell_num"
].map(
    lambda x:
        f"epSanyo{int(x):03d}"
)

history = (
    history
    .sort_values(
        [
            "canonical_cell_id",
            "zyk_ts_last",
            "cycle_number_pre_filter",
        ]
    )
    .reset_index(
        drop=True
    )
)

history[
    "usable_history_rank_zero_based"
] = (
    history
    .groupby(
        "canonical_cell_id",
        sort=False,
    )
    .cumcount()
)

history_out_cols = [
    "canonical_cell_id",
    "raw_cell_id",
    "usable_history_rank_zero_based",
    "cycle_number_pre_filter",
    "cycle_number_after_filter",
    "zyk_ts_last",
    "raw_cycle_key",
    "raw_cycle_position_zero_based",
    "source_zip_first",
    "source_zip_last",
    "feature_contract_version",
    "kernel_sha256",
] + FEATURE_COLUMNS

history_out = history[
    history_out_cols
].copy()

assert len(
    history_out
) == 104940

assert history_out[
    "canonical_cell_id"
].nunique() == 48

X = (
    history_out[
        FEATURE_COLUMNS
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
    X
).all()

write_csv(
    history_out,
    RWTH_HISTORY_OUT,
)


# =============================================================================
# RWTH target table and causal-prefix alignment
# =============================================================================

legacy = pd.read_csv(
    AUDIT_DIR
    / "RWTH_target_to_ZYK_TS_causal_alignment.csv",
    low_memory=False,
)

assert len(
    legacy
) == 777

legacy[
    "_cell_num"
] = legacy[
    "canonical_cell"
].map(
    cell_num
)

legacy[
    "_target_ts"
] = legacy[
    "target_ts"
].map(
    ts_num
)


rl = (
    labels[
        labels["dataset"]
        == "RWTH"
    ]
    .copy()
)

rl[
    "_cell_num"
] = rl[
    "cell_id"
].map(
    cell_num
)


target = rl.merge(
    legacy[
        [
            "_cell_num",
            "target_observation_index",
            "_target_ts",
            "target_life_coordinate",
            "prior_zyk_cycle_count",
            "latest_prior_zyk_cycle",
            "latest_prior_zyk_ts",
            "alignment_rule",
        ]
    ],
    left_on=[
        "_cell_num",
        "observation_index",
    ],
    right_on=[
        "_cell_num",
        "target_observation_index",
    ],
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert len(
    target
) == 777

assert (
    target["_merge"]
    == "both"
).all()

assert np.allclose(
    pd.to_numeric(
        target[
            "life_coordinate"
        ],
        errors="raise",
    ),
    pd.to_numeric(
        target[
            "target_life_coordinate"
        ],
        errors="raise",
    ),
    rtol=0.0,
    atol=1e-10,
)


target_rows = []

legacy_count_match = 0
changed_history_count = 0
changed_latest = 0
removed_history_total = 0
max_removed = 0


for cell, tg in target.groupby(
    "_cell_num",
    sort=True,
):
    all_cell = (
        rz[
            rz[
                "_cell_num"
            ] == cell
        ]
        .sort_values(
            [
                "zyk_ts_last",
                "cycle_number_pre_filter",
            ]
        )
    )

    old_pool = (
        all_cell[
            all_cell[
                "_qd_eligible"
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    new_pool = (
        all_cell[
            all_cell[
                "_usable"
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    old_ts = pd.to_numeric(
        old_pool[
            "zyk_ts_last"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    new_ts = pd.to_numeric(
        new_pool[
            "zyk_ts_last"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
    )

    for _, tr in tg.iterrows():
        target_ts = int(
            tr[
                "_target_ts"
            ]
        )

        old_n = int(
            np.searchsorted(
                old_ts,
                target_ts,
                side="left",
            )
        )

        new_n = int(
            np.searchsorted(
                new_ts,
                target_ts,
                side="left",
            )
        )

        legacy_n = int(
            tr[
                "prior_zyk_cycle_count"
            ]
        )

        assert old_n == legacy_n

        legacy_count_match += 1

        old_latest = (
            old_pool.iloc[
                old_n - 1
            ]
            if old_n
            else None
        )

        new_latest = (
            new_pool.iloc[
                new_n - 1
            ]
            if new_n
            else None
        )

        if old_n != new_n:
            changed_history_count += 1

        removed = (
            old_n
            - new_n
        )

        removed_history_total += (
            removed
        )

        max_removed = max(
            max_removed,
            removed,
        )

        old_latest_after = (
            int(
                old_latest[
                    "cycle_number_after_filter"
                ]
            )
            if old_latest
            is not None
            else np.nan
        )

        new_latest_after = (
            int(
                new_latest[
                    "cycle_number_after_filter"
                ]
            )
            if new_latest
            is not None
            else np.nan
        )

        old_cmp = (
            -1
            if pd.isna(
                old_latest_after
            )
            else int(
                old_latest_after
            )
        )

        new_cmp = (
            -1
            if pd.isna(
                new_latest_after
            )
            else int(
                new_latest_after
            )
        )

        if old_cmp != new_cmp:
            changed_latest += 1

        if new_latest is not None:
            assert int(
                new_latest[
                    "zyk_ts_last"
                ]
            ) < target_ts

        target_rows.append({
            "dataset":
                "RWTH",

            "cohort":
                tr["cohort"],

            "cell_id":
                tr["cell_id"],

            "dataset_cell_key":
                tr[
                    "dataset_cell_key"
                ],

            "split_group":
                tr[
                    "split_group"
                ],

            "observation_index":
                int(
                    tr[
                        "observation_index"
                    ]
                ),

            "life_coordinate":
                tr[
                    "life_coordinate"
                ],

            "life_coordinate_name":
                tr[
                    "life_coordinate_name"
                ],

            "capacity_Ah":
                tr[
                    "capacity_Ah"
                ],

            "SOH":
                tr["SOH"],

            "RUL_observations":
                tr[
                    "RUL_observations"
                ],

            "native_RUL":
                tr[
                    "native_RUL"
                ],

            "target_source":
                tr[
                    "target_source"
                ],

            "censoring_status":
                tr[
                    "censoring_status"
                ],

            "target_ts":
                target_ts,

            "legacy_qd_clean_history_count":
                old_n,

            "usable_history_count":
                new_n,

            "removed_structural_boundary_cycles":
                removed,

            "usable_history_start_rank_zero_based":
                0,

            "usable_history_end_rank_exclusive":
                new_n,

            "has_usable_sensor_history":
                bool(
                    new_n > 0
                ),

            "latest_usable_cycle_pre_filter":
                (
                    int(
                        new_latest[
                            "cycle_number_pre_filter"
                        ]
                    )
                    if new_latest
                    is not None
                    else np.nan
                ),

            "latest_usable_cycle_after_filter":
                (
                    int(
                        new_latest[
                            "cycle_number_after_filter"
                        ]
                    )
                    if new_latest
                    is not None
                    else np.nan
                ),

            "latest_usable_zyk_ts":
                (
                    int(
                        new_latest[
                            "zyk_ts_last"
                        ]
                    )
                    if new_latest
                    is not None
                    else np.nan
                ),

            "causal_alignment_rule":
                "zyk_ts_last < target_ts",

            "history_representation":
                (
                    "prefix of "
                    "RWTH_usable_history_cycles "
                    "for same canonical cell"
                ),
        })


rwth_targets = pd.DataFrame(
    target_rows
)

assert len(
    rwth_targets
) == 777

assert rwth_targets[
    "cell_id"
].nunique() == 48

assert legacy_count_match == 777
assert changed_history_count == 390
assert changed_latest == 0
assert removed_history_total == 609
assert max_removed == 8

assert int(
    (
        rwth_targets[
            "usable_history_count"
        ] == 0
    ).sum()
) == 48

assert int(
    rwth_targets[
        "has_usable_sensor_history"
    ].sum()
) == 729

assert int(
    (
        rwth_targets[
            "removed_structural_boundary_cycles"
        ] > 0
    ).sum()
) == 390

write_csv(
    rwth_targets,
    RWTH_TARGET_OUT,
)


# =============================================================================
# Final guards / manifest
# =============================================================================

feature_valid_counts = (
    supervised
    .groupby(
        "dataset"
    )[
        "feature_valid"
    ]
    .sum()
    .astype(int)
    .to_dict()
)

feature_available_counts = (
    supervised
    .groupby(
        "dataset"
    )[
        "feature_available"
    ]
    .sum()
    .astype(int)
    .to_dict()
)

supervised_eligible_counts = (
    supervised
    .groupby(
        "dataset"
    )[
        "supervised_feature_eligible"
    ]
    .sum()
    .astype(int)
    .to_dict()
)

soh_eligible_counts = (
    supervised
    .groupby(
        "dataset"
    )[
        "soh_task_eligible"
    ]
    .sum()
    .astype(int)
    .to_dict()
)

rul_eligible_counts = (
    supervised
    .groupby(
        "dataset"
    )[
        "rul_task_eligible"
    ]
    .sum()
    .astype(int)
    .to_dict()
)


assert feature_available_counts == {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140174,
}

assert feature_valid_counts == {
    "HUST": 146122,
    "CALCE": 14298,
    "MATR": 140133,
}


target_like_feature_names = [
    c
    for c in FEATURE_COLUMNS
    if any(
        token in c.lower()
        for token in (
            "capacity",
            "soh",
            "rul",
            "eol",
            "cycle",
            "target",
            "qd",
            "dq",
        )
    )
]

assert target_like_feature_names == []


manifest = {
    "builder_contract_version":
        BUILDER_CONTRACT_VERSION,

    "feature_contract_version":
        FEATURE_CONTRACT_VERSION,

    "label_backbone_sha256":
        sha256_file(
            LABEL_BACKBONE
        ),

    "input_feature_sha256":
        {
            dataset:
                EXPECTED_FEATURE_HASHES[
                    dataset
                ]
            for dataset
            in (
                "HUST",
                "CALCE",
                "MATR",
                "RWTH",
            )
        },

    "primary_feature_columns":
        FEATURE_COLUMNS,

    "target_like_primary_feature_names":
        target_like_feature_names,

    "split_applied":
        False,

    "scaler_applied":
        False,

    "supervised_table_rows":
        len(
            supervised
        ),

    "supervised_rows_by_dataset":
        supervised[
            "dataset"
        ].value_counts().to_dict(),

    "feature_available_by_dataset":
        feature_available_counts,

    "feature_valid_by_dataset":
        feature_valid_counts,

    "supervised_feature_eligible_by_dataset":
        supervised_eligible_counts,

    "soh_task_eligible_by_dataset":
        soh_eligible_counts,

    "rul_task_eligible_by_dataset":
        rul_eligible_counts,

    "matr": {
        "canonical_label_rows":
            140175,

        "feature_matched_rows":
            140174,

        "feature_valid_matched_rows":
            140133,

        "mapped_invalid_rows":
            41,

        "b4c31_featureless_final_label_rows":
            1,

        "excluded_raw_cells":
            excluded_cells,

        "excluded_raw_cycle_rows":
            14237,

        "invalid_rows_in_excluded_raw_cells":
            5,

        "continuations":
            {
                k: {
                    "donor":
                        v[0],

                    "donor_cycle_count":
                        v[1],
                }
                for k, v in
                CONTINUATIONS.items()
            },
    },

    "rwth": {
        "structural_cycles":
            105663,

        "qd_eligible_cycles":
            105006,

        "feature_valid_cycles":
            105017,

        "usable_qd_clean_contiguous_cycles":
            104940,

        "qd_clean_but_structurally_invalid_cycles":
            66,

        "sparse_targets":
            777,

        "targets_with_usable_history":
            729,

        "zero_history_targets":
            48,

        "targets_whose_history_count_changes_vs_legacy":
            390,

        "targets_whose_latest_prior_cycle_changes_vs_legacy":
            0,

        "total_removed_boundary_history_incidences":
            609,

        "max_removed_cycles_from_one_target_history":
            8,

        "causal_rule":
            "zyk_ts_last < target_ts",

        "history_storage":
            (
                "per-cell chronologically sorted usable-cycle "
                "table plus target-specific prefix end index"
            ),
    },

    "outputs": {
        "canonical_cycle_supervised_table": {
            "path":
                str(
                    SUPERVISED_OUT
                    .relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    SUPERVISED_OUT
                ),
        },

        "RWTH_usable_history_cycles": {
            "path":
                str(
                    RWTH_HISTORY_OUT
                    .relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    RWTH_HISTORY_OUT
                ),
        },

        "RWTH_sparse_target_table": {
            "path":
                str(
                    RWTH_TARGET_OUT
                    .relative_to(
                        BASE
                    )
                ),

            "sha256":
                sha256_file(
                    RWTH_TARGET_OUT
                ),
        },
    },
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 78)
print("CANONICAL TASK ALIGNMENT BUILDER")
print("=" * 78)

print(
    "supervised rows:",
    len(supervised)
)

print(
    "rows by dataset:",
    supervised[
        "dataset"
    ].value_counts().to_dict()
)

print(
    "feature available:",
    feature_available_counts
)

print(
    "feature valid:",
    feature_valid_counts
)

print(
    "supervised eligible:",
    supervised_eligible_counts
)

print(
    "SOH eligible:",
    soh_eligible_counts
)

print(
    "RUL eligible:",
    rul_eligible_counts
)

print()
print(
    "MATR matched / valid / featureless:",
    "140174 / 140133 / 1"
)

print()
print(
    "RWTH usable history cycles:",
    len(history_out)
)

print(
    "RWTH targets / with-history / zero-history:",
    "777 / 729 / 48"
)

print(
    "RWTH changed history counts:",
    changed_history_count
)

print(
    "RWTH changed latest prior cycles:",
    changed_latest
)

print(
    "RWTH removed history incidences:",
    removed_history_total
)

print()
print(
    "supervised SHA256:",
    sha256_file(
        SUPERVISED_OUT
    )
)

print(
    "RWTH history SHA256:",
    sha256_file(
        RWTH_HISTORY_OUT
    )
)

print(
    "RWTH targets SHA256:",
    sha256_file(
        RWTH_TARGET_OUT
    )
)

print(
    "manifest SHA256:",
    sha256_file(
        MANIFEST_OUT
    )
)

print()
print(
    "split applied: False"
)

print(
    "scaler applied: False"
)

print(
    "target-like primary features: 0"
)

print()
print(
    "CANONICAL_TASK_ALIGNMENT_BUILDER_PASS=True"
)
