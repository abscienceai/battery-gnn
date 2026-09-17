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

CANON = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "canonical_tasks"
)

SPLITDIR = (
    BASE
    / "data"
    / "processed"
    / "revision_2026"
    / "splits_and_scalers"
)

SUP = (
    CANON
    / "canonical_cycle_supervised_table.csv"
)

RHIST = (
    CANON
    / "RWTH_usable_history_cycles.csv"
)

RTGT = (
    CANON
    / "RWTH_sparse_target_table.csv"
)

ASSIGN = (
    SPLITDIR
    / "cell_split_assignments.csv"
)

PARAMS = (
    SPLITDIR
    / "train_only_scaler_parameters.csv"
)

SUMMARY = (
    SPLITDIR
    / "train_only_scaler_fit_summary.csv"
)

MANIFEST = (
    SPLITDIR
    / "grouped_split_scaler_manifest.json"
)


EXPECTED_HASHES = {
    "assignments":
        "914c8b90131f496ca396f63e0502147ca99848fdfd4b2c6b2a5887225b47d055",

    "parameters":
        "b9708b9321b521b87bdd05f969558134d7e0148285ef7c50f648c2028cd86de1",

    "summary":
        "e17ffa78cbec4dcbb032951e2030f99860c27cfed7da7510ab0d147cc8f25ab9",

    "manifest":
        "91aee27fa332e053454c4f2300c367dc363b938e5ab3d12513372370ec4f2238",
}


INPUT_HASHES = {
    "supervised":
        "49a7b7629cf9ee3f8ce0a7e3c52f2807878b752b4c9a30a876285fb891b00c14",

    "rwth_history":
        "4e8dcc9d80e0284e3c6cb9f912505fbff066efff052d3e08996da4028e3a0faa",

    "rwth_targets":
        "6346566441b265e9f5d328527ebeb58a0c9dfb2d18ba70ab72ea7b9f2808eb54",
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

SEEDS = [0, 1, 2, 3, 4]

SPLITS = [
    "train",
    "validation",
    "test",
]

RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

CONTRACT = (
    "grouped_dataset_censor_stratified_v1"
)


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


def stable_u64(text):
    raw = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        raw[:8],
        byteorder="little",
        signed=False,
    )


def allocation_counts(n):
    exact = np.array(
        [
            RATIOS["train"] * n,
            RATIOS["validation"] * n,
            RATIOS["test"] * n,
        ],
        dtype=float,
    )

    counts = np.floor(
        exact
    ).astype(int)

    remaining = int(
        n - counts.sum()
    )

    fractions = (
        exact - counts
    )

    order = sorted(
        range(3),
        key=lambda i: (
            -fractions[i],
            i,
        ),
    )

    for i in order[:remaining]:
        counts[i] += 1

    out = {
        split:
            int(n_)
        for split, n_
        in zip(
            SPLITS,
            counts,
        )
    }

    assert sum(
        out.values()
    ) == n

    return out


def one_value(g, col):
    values = (
        g[col]
        .drop_duplicates()
        .tolist()
    )

    assert len(values) == 1, (
        col,
        values,
    )

    return values[0]


def cell_profile(df):
    rows = []

    for split_group, g in df.groupby(
        "split_group",
        sort=True,
    ):
        dataset = str(
            one_value(
                g,
                "dataset",
            )
        )

        censor = str(
            one_value(
                g,
                "censoring_status",
            )
        )

        rows.append({
            "split_group":
                str(split_group),

            "dataset":
                dataset,

            "cell_id":
                str(
                    one_value(
                        g,
                        "cell_id",
                    )
                ),

            "cohort":
                str(
                    one_value(
                        g,
                        "cohort",
                    )
                ),

            "censoring_status":
                censor,

            "stratum":
                dataset
                + "::"
                + censor,
        })

    return pd.DataFrame(
        rows
    )


def independent_assignments(profile):
    rows = []

    for seed in SEEDS:

        for stratum, g in profile.groupby(
            "stratum",
            sort=True,
        ):
            groups = np.array(
                sorted(
                    g[
                        "split_group"
                    ].astype(str)
                ),
                dtype=object,
            )

            counts = allocation_counts(
                len(groups)
            )

            h = stable_u64(
                stratum
            )

            ss = np.random.SeedSequence(
                [
                    int(seed),
                    int(
                        h
                        & 0xFFFFFFFF
                    ),
                    int(
                        (
                            h >> 32
                        )
                        & 0xFFFFFFFF
                    ),
                ]
            )

            rng = np.random.default_rng(
                ss
            )

            shuffled = groups[
                rng.permutation(
                    len(groups)
                )
            ]

            start = 0

            for split in SPLITS:
                end = (
                    start
                    + counts[split]
                )

                for split_group in (
                    shuffled[
                        start:end
                    ]
                ):
                    r = g[
                        g[
                            "split_group"
                        ].astype(str)
                        == str(
                            split_group
                        )
                    ].iloc[0]

                    rows.append({
                        "split_seed":
                            seed,

                        "split_group":
                            str(
                                split_group
                            ),

                        "dataset":
                            str(
                                r["dataset"]
                            ),

                        "cell_id":
                            str(
                                r["cell_id"]
                            ),

                        "cohort":
                            str(
                                r["cohort"]
                            ),

                        "censoring_status":
                            str(
                                r[
                                    "censoring_status"
                                ]
                            ),

                        "stratum":
                            stratum,

                        "split":
                            split,

                        "split_contract_version":
                            CONTRACT,
                    })

                start = end

    return pd.DataFrame(
        rows
    )


def fit_scaler(df, mask):
    x = (
        df.loc[
            mask,
            FEATURES,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    assert len(x) > 0
    assert np.isfinite(
        x
    ).all()

    mean = np.mean(
        x,
        axis=0,
        dtype=np.float64,
    )

    var = np.var(
        x,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    scale = np.sqrt(
        var
    )

    zero = (
        scale == 0
    )

    scale = scale.copy()
    scale[
        zero
    ] = 1.0

    return (
        len(x),
        mean,
        var,
        scale,
        zero,
    )


print("=" * 80)
print("GROUPED SPLIT + TRAIN-ONLY SCALER INDEPENDENT AUDIT")
print("=" * 80)


# =========================================================================
# 1. Immutable inputs / outputs
# =========================================================================

assert sha256_file(
    SUP
) == INPUT_HASHES[
    "supervised"
]

assert sha256_file(
    RHIST
) == INPUT_HASHES[
    "rwth_history"
]

assert sha256_file(
    RTGT
) == INPUT_HASHES[
    "rwth_targets"
]


observed_hashes = {
    "assignments":
        sha256_file(
            ASSIGN
        ),

    "parameters":
        sha256_file(
            PARAMS
        ),

    "summary":
        sha256_file(
            SUMMARY
        ),

    "manifest":
        sha256_file(
            MANIFEST
        ),
}

assert (
    observed_hashes
    == EXPECTED_HASHES
)

print(
    "input hashes exact: True"
)

print(
    "output hashes exact: True"
)


# =========================================================================
# 2. Load cell metadata and rebuild profile independently
# =========================================================================

sup = pd.read_csv(
    SUP,
    usecols=[
        "dataset",
        "cohort",
        "cell_id",
        "split_group",
        "censoring_status",
        "supervised_feature_eligible",
    ] + FEATURES,
    low_memory=False,
)

rt = pd.read_csv(
    RTGT,
    usecols=[
        "dataset",
        "cohort",
        "cell_id",
        "split_group",
        "censoring_status",
    ],
    low_memory=False,
)

rh = pd.read_csv(
    RHIST,
    usecols=[
        "canonical_cell_id",
    ] + FEATURES,
    low_memory=False,
)


sp = cell_profile(
    sup[
        [
            "dataset",
            "cohort",
            "cell_id",
            "split_group",
            "censoring_status",
        ]
    ]
)

rp = cell_profile(
    rt[
        [
            "dataset",
            "cohort",
            "cell_id",
            "split_group",
            "censoring_status",
        ]
    ]
)

profile = pd.concat(
    [
        sp,
        rp,
    ],
    ignore_index=True,
)

assert len(profile) == 307
assert (
    profile[
        "split_group"
    ].nunique()
    == 307
)


censor_counts = (
    profile.groupby(
        [
            "dataset",
            "censoring_status",
        ]
    )
    .size()
    .to_dict()
)

assert censor_counts == {
    (
        "CALCE",
        "observed_event",
    ): 13,

    (
        "HUST",
        "observed_event",
    ): 77,

    (
        "MATR",
        "observed_event",
    ): 88,

    (
        "MATR",
        "right_censored_len_plus_1",
    ): 81,

    (
        "RWTH",
        "observed_event",
    ): 48,
}


print(
    "independent cell profile:",
    "307 / 307"
)


# =========================================================================
# 3. Independently regenerate all five split assignments
# =========================================================================

expected_assign = (
    independent_assignments(
        profile
    )
)

observed_assign = pd.read_csv(
    ASSIGN,
    low_memory=False,
)

assert len(
    observed_assign
) == 1535

assert len(
    expected_assign
) == 1535


compare_cols = [
    "split_seed",
    "split_group",
    "dataset",
    "cell_id",
    "cohort",
    "censoring_status",
    "stratum",
    "split",
    "split_contract_version",
]


expected_cmp = (
    expected_assign[
        compare_cols
    ]
    .sort_values(
        [
            "split_seed",
            "split_group",
        ]
    )
    .reset_index(
        drop=True
    )
)

observed_cmp = (
    observed_assign[
        compare_cols
    ]
    .sort_values(
        [
            "split_seed",
            "split_group",
        ]
    )
    .reset_index(
        drop=True
    )
)


assert expected_cmp.equals(
    observed_cmp
)

print(
    "independent assignment reproduction:",
    "1535 / 1535"
)


# =========================================================================
# 4. Cell-disjointness and split counts for every seed
# =========================================================================

expected_dataset_counts = {
    "HUST": {
        "train": 54,
        "validation": 12,
        "test": 11,
    },

    "CALCE": {
        "train": 9,
        "validation": 2,
        "test": 2,
    },

    "MATR": {
        "train": 119,
        "validation": 25,
        "test": 25,
    },

    "RWTH": {
        "train": 34,
        "validation": 7,
        "test": 7,
    },
}


for seed in SEEDS:

    a = observed_assign[
        observed_assign[
            "split_seed"
        ] == seed
    ]

    assert len(a) == 307
    assert (
        a[
            "split_group"
        ].nunique()
        == 307
    )

    groups = {}

    for split in SPLITS:
        groups[split] = set(
            a.loc[
                a["split"]
                == split,
                "split_group",
            ].astype(str)
        )

    assert groups[
        "train"
    ].isdisjoint(
        groups[
            "validation"
        ]
    )

    assert groups[
        "train"
    ].isdisjoint(
        groups[
            "test"
        ]
    )

    assert groups[
        "validation"
    ].isdisjoint(
        groups[
            "test"
        ]
    )

    assert len(
        groups["train"]
        | groups["validation"]
        | groups["test"]
    ) == 307

    for dataset, expected in (
        expected_dataset_counts.items()
    ):
        got = (
            a[
                a[
                    "dataset"
                ] == dataset
            ][
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        assert got == expected


    matr = a[
        a[
            "dataset"
        ] == "MATR"
    ]

    assert (
        matr[
            matr[
                "censoring_status"
            ] == "observed_event"
        ][
            "split"
        ]
        .value_counts()
        .to_dict()
        == {
            "train": 62,
            "validation": 13,
            "test": 13,
        }
    )

    assert (
        matr[
            matr[
                "censoring_status"
            ]
            == "right_censored_len_plus_1"
        ][
            "split"
        ]
        .value_counts()
        .to_dict()
        == {
            "train": 57,
            "validation": 12,
            "test": 12,
        }
    )


print(
    "all five seeds cell-disjoint: True"
)

print(
    "dataset split counts exact for all seeds: True"
)

print(
    "MATR censoring stratification exact for all seeds: True"
)


# =========================================================================
# 5. Independently reconstruct all 25 scaler fits
# =========================================================================

sup["_eligible"] = as_bool(
    sup[
        "supervised_feature_eligible"
    ]
)

rh["split_group"] = (
    "RWTH::"
    + rh[
        "canonical_cell_id"
    ].astype(str)
)


expected_param_rows = []
expected_summary_rows = []


for seed in SEEDS:

    a = observed_assign[
        observed_assign[
            "split_seed"
        ] == seed
    ]

    train_groups = set(
        a.loc[
            a[
                "split"
            ] == "train",
            "split_group",
        ].astype(str)
    )

    validation_groups = set(
        a.loc[
            a[
                "split"
            ] == "validation",
            "split_group",
        ].astype(str)
    )

    test_groups = set(
        a.loc[
            a[
                "split"
            ] == "test",
            "split_group",
        ].astype(str)
    )

    assert train_groups.isdisjoint(
        validation_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )


    scopes = []


    pooled_mask = (
        sup["_eligible"]
        &
        sup[
            "split_group"
        ].astype(str).isin(
            train_groups
        )
    )

    scopes.append(
        (
            "pooled_supervised",
            sup,
            pooled_mask,
        )
    )


    for dataset in (
        "HUST",
        "CALCE",
        "MATR",
    ):
        mask = (
            sup["_eligible"]
            &
            (
                sup["dataset"]
                == dataset
            )
            &
            sup[
                "split_group"
            ].astype(str).isin(
                train_groups
            )
        )

        scopes.append(
            (
                dataset,
                sup,
                mask,
            )
        )


    rwth_train = set(
        a.loc[
            (
                a["dataset"]
                == "RWTH"
            )
            &
            (
                a["split"]
                == "train"
            ),
            "split_group",
        ].astype(str)
    )

    rwth_mask = (
        rh[
            "split_group"
        ].astype(str).isin(
            rwth_train
        )
    )

    scopes.append(
        (
            "RWTH_history",
            rh,
            rwth_mask,
        )
    )


    for scope, frame, mask in scopes:

        n, mean, var, scale, zero = (
            fit_scaler(
                frame,
                mask,
            )
        )

        used_groups = set(
            frame.loc[
                mask,
                "split_group",
            ].astype(str)
        )

        assert used_groups.issubset(
            train_groups
        )

        assert used_groups.isdisjoint(
            validation_groups
        )

        assert used_groups.isdisjoint(
            test_groups
        )

        expected_summary_rows.append({
            "split_seed":
                seed,

            "scaler_scope":
                scope,

            "n_train_rows":
                n,

            "n_train_groups":
                len(
                    used_groups
                ),

            "zero_variance_features":
                int(
                    zero.sum()
                ),

            "fit_split":
                "train_only",
        })

        for j, feature in enumerate(
            FEATURES
        ):
            expected_param_rows.append({
                "split_seed":
                    seed,

                "scaler_scope":
                    scope,

                "feature":
                    feature,

                "mean":
                    float(
                        mean[j]
                    ),

                "variance":
                    float(
                        var[j]
                    ),

                "scale":
                    float(
                        scale[j]
                    ),

                "zero_variance":
                    bool(
                        zero[j]
                    ),

                "n_train_rows":
                    n,

                "fit_split":
                    "train_only",
            })


expected_params = pd.DataFrame(
    expected_param_rows
)

expected_summary = pd.DataFrame(
    expected_summary_rows
)

observed_params = pd.read_csv(
    PARAMS,
    low_memory=False,
)

observed_summary = pd.read_csv(
    SUMMARY,
    low_memory=False,
)


assert len(
    expected_params
) == 500

assert len(
    observed_params
) == 500

assert len(
    expected_summary
) == 25

assert len(
    observed_summary
) == 25


key = [
    "split_seed",
    "scaler_scope",
    "feature",
]

ep = (
    expected_params
    .sort_values(
        key
    )
    .reset_index(
        drop=True
    )
)

op = (
    observed_params
    .sort_values(
        key
    )
    .reset_index(
        drop=True
    )
)

assert (
    ep[key].astype(str)
    .equals(
        op[key].astype(str)
    )
)


for col in (
    "n_train_rows",
    "fit_split",
):
    assert (
        ep[col]
        .astype(str)
        .equals(
            op[col]
            .astype(str)
        )
    )


assert np.array_equal(
    ep[
        "zero_variance"
    ].astype(bool).to_numpy(),
    as_bool(
        op[
            "zero_variance"
        ]
    ).to_numpy(),
)


mean_delta = np.abs(
    ep["mean"].to_numpy(
        dtype=float
    )
    -
    op["mean"].to_numpy(
        dtype=float
    )
)

var_delta = np.abs(
    ep["variance"].to_numpy(
        dtype=float
    )
    -
    op["variance"].to_numpy(
        dtype=float
    )
)

scale_delta = np.abs(
    ep["scale"].to_numpy(
        dtype=float
    )
    -
    op["scale"].to_numpy(
        dtype=float
    )
)


assert np.allclose(
    ep["mean"],
    op["mean"],
    rtol=5e-15,
    atol=5e-15,
)

assert np.allclose(
    ep["variance"],
    op["variance"],
    rtol=5e-15,
    atol=5e-15,
)

assert np.allclose(
    ep["scale"],
    op["scale"],
    rtol=5e-15,
    atol=5e-15,
)


summary_key = [
    "split_seed",
    "scaler_scope",
]

es = (
    expected_summary
    .sort_values(
        summary_key
    )
    .reset_index(
        drop=True
    )
)

os_ = (
    observed_summary
    .sort_values(
        summary_key
    )
    .reset_index(
        drop=True
    )
)


assert (
    es[
        summary_key
    ].astype(str)
    .equals(
        os_[
            summary_key
        ].astype(str)
    )
)

for col in (
    "n_train_rows",
    "n_train_groups",
    "zero_variance_features",
    "fit_split",
):
    assert (
        es[col]
        .astype(str)
        .equals(
            os_[col]
            .astype(str)
        )
    )


assert int(
    observed_summary[
        "zero_variance_features"
    ].sum()
) == 0


print(
    "independent scaler fits reproduced:",
    "25 / 25"
)

print(
    "independent scaler parameters reproduced:",
    "500 / 500"
)

print(
    "max mean abs delta:",
    f"{float(mean_delta.max()):.3e}"
)

print(
    "max variance abs delta:",
    f"{float(var_delta.max()):.3e}"
)

print(
    "max scale abs delta:",
    f"{float(scale_delta.max()):.3e}"
)

print(
    "zero-variance features across fits:",
    0
)

print(
    "validation/test groups entering scaler fits:",
    0
)


# =========================================================================
# 6. Manifest policy checks
# =========================================================================

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

assert manifest[
    "contract_version"
] == CONTRACT

assert manifest[
    "split_seeds"
] == SEEDS

assert manifest[
    "total_unique_groups"
] == 307

assert manifest[
    "grouping_unit"
] == (
    "split_group / cell"
)

assert manifest[
    "scaler_fit_policy"
] == (
    "train groups only"
)

assert manifest[
    "validation_statistics_used_for_scaler_fit"
] is False

assert manifest[
    "test_statistics_used_for_scaler_fit"
] is False

assert manifest[
    "scaled_data_materialized"
] is False

assert manifest[
    "lodo_scalers_created"
] is False

assert manifest[
    "primary_features"
] == FEATURES


print()
print(
    "grouping unit: cell"
)

print(
    "scaler fit policy: train groups only"
)

print(
    "validation statistics used: False"
)

print(
    "test statistics used: False"
)

print(
    "LODO scalers created: False"
)

print(
    "scaled data materialized: False"
)

print()
print(
    "GROUPED_SPLIT_SCALER_AUDIT_PASS=True"
)
